<#
.SYNOPSIS
  push_run.ps1 — push a local payload to the Windows SLM runner, run it, and pull outputs.

.DESCRIPTION
  Usage:
    .\push_run.ps1 <payload_file> [-HoldOn] [-Png] [-PngAnaly]
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Payload,

    [switch]$HoldOn,
    [switch]$Png,
    [switch]$PngAnaly
)

$ErrorActionPreference = "Stop"

# 1. 检查参数冲突和文件存在
if ($Png -and $PngAnaly) {
    Write-Error "ERROR: -Png and -PngAnaly are mutually exclusive"
    exit 1
}

if (-not (Test-Path $Payload -PathType Leaf)) {
    Write-Error "ERROR: payload file not found: $Payload"
    exit 1
}

# 统一路径分隔符为 / 以方便解析
$PayloadFS = $Payload -replace "\\", "/"
if ($PayloadFS -notmatch "^payload/") {
    Write-Error "ERROR: payload must be under payload/ (got: $Payload)"
    exit 1
}

# 2. 解析路径变量
$Rel = $PayloadFS -replace "^payload/", ""          # e.g. sheet/foo_payload.npz
$Subdir = Split-Path $Rel -Parent                   # e.g. sheet
$Filename = Split-Path $Rel -Leaf                   # e.g. foo_payload.npz
$Base = $Filename -replace "_payload\.npz$", ""     # e.g. foo
$Params = "payload/$Subdir/${Base}_params.json"
$SubdirBS = $Subdir -replace "/", "\"               # e.g. sheet (如果有多级会变成 a\b)

# 3. 读取配置文件 (hamamatsu_test_config.json)
$ConfigFile = "hamamatsu_test_config.json"
$Config = @{}
if (Test-Path $ConfigFile) {
    try {
        $json = Get-Content $ConfigFile -Raw | ConvertFrom-Json
        if ($null -ne $json.windows_local) {
            $Config = $json.windows_local
        }
    } catch {
        Write-Warning "Failed to parse $ConfigFile, using defaults."
    }
}

$ServerIp   = if ($Config.host) { $Config.host } else { "192.168.50.35" }
$Port       = if ($Config.port) { $Config.port } else { 22 }
$SshUser    = if ($Config.user) { $Config.user } else { "Galileo" }
$RemoteBase = if ($Config.remote_base) { $Config.remote_base } else { "C:/Users/Galileo/SLMengineer/windows_runner" }
$RemoteRepo = if ($Config.remote_repo) { $Config.remote_repo } else { "C:/Users/Galileo/SLMengineer" }

# 目标连接字符串
$Target = "${SshUser}@${ServerIp}"
$SshCmd = "ssh"
$ScpCmd = "scp"

# 远程路径处理 (FS: Forward Slash 用于 scp; BS: Backslash 用于 windows cmd)
$WinRunnerBS = $RemoteBase -replace "/", "\"
$RemoteRepoBS = $RemoteRepo -replace "/", "\"
$RemoteIncomingBS = "${WinRunnerBS}\incoming\${SubdirBS}"
$RemoteIncomingFS = "${RemoteBase}/incoming/${Subdir}"
$RemoteDataBS = "${WinRunnerBS}\data\${SubdirBS}"
$RemoteDataFS = "${RemoteBase}/data/${Subdir}"
$LocalDataDir = "data/${Subdir}"

$RunPrefix = "${SubdirBS}\${Base}"

# --- 开始执行流程 ---
Write-Host "[1/4] Ensuring remote ${RemoteIncomingBS} exists..."
& $SshCmd -p $Port $Target "if not exist `"${RemoteIncomingBS}`" mkdir `"${RemoteIncomingBS}`""

Write-Host "[2/4] Pushing payload..."
& $ScpCmd -P $Port $Payload "${Target}:${RemoteIncomingFS}/"
if (Test-Path $Params) {
    & $ScpCmd -P $Port $Params "${Target}:${RemoteIncomingFS}/"
    Write-Host "  pushed $Filename + $(Split-Path $Params -Leaf)"
} else {
    Write-Host "  pushed $Filename (no params.json sibling)"
}

# 读取附加参数
$RunnerArgs = ""
if (Test-Path $Params) {
    try {
        $pData = Get-Content $Params -Raw | ConvertFrom-Json
        if ($null -ne $pData.runner_defaults) {
            if ($null -ne $pData.runner_defaults.etime_us) { $RunnerArgs += " --etime-us " + $pData.runner_defaults.etime_us }
            if ($null -ne $pData.runner_defaults.n_avg) { $RunnerArgs += " --n-avg " + $pData.runner_defaults.n_avg }
        }
    } catch {}
}

$HoldFlagStr = if ($HoldOn) { " --hold-on" } else { "" }
$TriggerMsg = "[3/4] Triggering slmrun.bat"
if ($HoldOn) { $TriggerMsg += " (--hold-on)" }
if ($RunnerArgs) { $TriggerMsg += " (args:$RunnerArgs)" }
Write-Host "$TriggerMsg..."

# 执行远程 Batch
& $SshCmd -p $Port $Target "cd /d `"${WinRunnerBS}`" && slmrun.bat --payload `"incoming\${SubdirBS}\${Filename}`" --output-prefix `"${RunPrefix}`"${RunnerArgs}${HoldFlagStr}"

if ($HoldOn) {
    Write-Host "[4/4] hold-on mode: skipping pull."
    Write-Host "Done."
    exit 0
}

# 输出处理分支
$PullFiles = @()

if ($Png) {
    Write-Host "[4/5] Rendering BMP→color-heatmap PNG on Windows..."
    
    # 构建 Python 脚本并通过管道发送给远程 uv run python -
    $PyScript = @"
import os, sys, numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data, base = sys.argv[1], sys.argv[2]
ok = True
for tag in ('before', 'after'):
    bmp = os.path.join(data, f'{base}_{tag}.bmp')
    png = os.path.join(data, f'{base}_{tag}.png')
    if not os.path.exists(bmp):
        print(f'  {tag}: bmp not found ({bmp})', file=sys.stderr)
        ok = False
        continue
    arr = np.asarray(Image.open(bmp).convert('L'), dtype=np.uint8)
    vmax = max(int(arr.max()), 1)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(arr, cmap='hot', vmin=0, vmax=vmax)
    ax.set_title(f'{base}_{tag}  shape={arr.shape}  min={int(arr.min())}  max={int(arr.max())}  mean={arr.mean():.1f}')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {tag}: {os.path.getsize(bmp)//1024}KB bmp -> {os.path.getsize(png)//1024}KB color png (hot cmap, vmax={vmax})')

sys.exit(0 if ok else 2)
"@
    
    # 使用 PowerShell 管道将字符串传给远端 stdin
    $PyScript | & $SshCmd -p $Port $Target "cd /d `"${RemoteRepoBS}`" && uv run python - `"${RemoteDataBS}`" `"${Base}`""

    Write-Host "[5/5] Pulling PNG frames + run.json into ${LocalDataDir}/ ..."
    $PullFiles = @("${Base}_before.png", "${Base}_after.png", "${Base}_run.json")

} elseif ($PngAnaly) {
    Write-Host "[4/5] Uploading analysis_sheet.py and running it on Windows..."
    & $ScpCmd -P $Port scripts/sheet/analysis_sheet.py "${Target}:${RemoteIncomingFS}/" | Out-Null
    
    $RemoteAnalysisWin = "${WinRunnerBS}\incoming\${SubdirBS}\analysis_sheet.py"
    $AfterWin  = "${RemoteDataBS}\${Base}_after.bmp"
    $BeforeWin = "${RemoteDataBS}\${Base}_before.bmp"
    $PlotWin   = "${RemoteDataBS}\${Base}_analysis.png"
    $ResultWin = "${RemoteDataBS}\${Base}_analysis.json"

    & $SshCmd -p $Port $Target "cd /d `"${RemoteRepoBS}`" && uv run python `"${RemoteAnalysisWin}`" --after `"${AfterWin}`" --before `"${BeforeWin}`" --plot `"${PlotWin}`" --result `"${ResultWin}`""

    Write-Host "[5/5] Pulling analysis PNG + JSON + run.json into ${LocalDataDir}/ ..."
    $PullFiles = @("${Base}_analysis.png", "${Base}_analysis.json", "${Base}_run.json")

} else {
    Write-Host "[4/4] Pulling BMP frames + run.json into ${LocalDataDir}/ ..."
    $PullFiles = @("${Base}_before.bmp", "${Base}_after.bmp", "${Base}_run.json")
}

# 执行 Pull 动作
if (-not (Test-Path $LocalDataDir)) {
    New-Item -ItemType Directory -Path $LocalDataDir | Out-Null
}

$PullOk = $true
foreach ($f in $PullFiles) {
    # 隐藏错误输出以模拟 2>/dev/null
    & $ScpCmd -P $Port "${Target}:${RemoteDataFS}/${f}" "${LocalDataDir}/" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ${LocalDataDir}/$f"
    } else {
        Write-Host "  MISSING: $f"
        $PullOk = $false
    }
}

Write-Host "Done."