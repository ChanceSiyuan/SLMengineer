@echo off
setlocal enabledelayedexpansion
rem slm_local.bat -- Windows-local dispatcher for the SLM runner.
rem Mirrors push_run.sh but skips ssh/scp: the payload is copied into
rem <runner_base>\incoming\<sub>\ on the same machine, slmrun.bat is
rem invoked locally, and BMP->PNG / analysis post-processing runs
rem locally too.  <runner_base> defaults to the windows_runner\ folder
rem of this repo (the .bat's own directory).  Use this on the Windows
rem lab box that has the SLM + camera physically attached.
rem
rem Usage (from the repo root, e.g. C:\Users\Galileo\SLMengineer):
rem   windows_runner\slm_local.bat <payload_file>                   (default: capture BMPs)
rem   windows_runner\slm_local.bat <payload_file> --hold-on         (display + hold, no capture)
rem   windows_runner\slm_local.bat <payload_file> --png             (also render BMPs to color PNGs)
rem   windows_runner\slm_local.bat <payload_file> --png-analy       (also run analysis_sheet.py)
rem
rem <payload_file> must live under payload\ (e.g. payload\sheet\testfile_sheet_payload.npz).

if "%~1"=="" goto :usage

set "PAYLOAD=%~1"
shift

set "HOLD_FLAG="
set "PNG_MODE="
set "ANALY_MODE="

:parseargs
if "%~1"=="" goto :doneargs
if /i "%~1"=="--hold-on"   ( set "HOLD_FLAG=--hold-on" & shift & goto :parseargs )
if /i "%~1"=="--png"       ( set "PNG_MODE=1"          & shift & goto :parseargs )
if /i "%~1"=="--png-analy" ( set "ANALY_MODE=1"        & shift & goto :parseargs )
echo ERROR: unknown arg: %~1
exit /b 1

:doneargs
if defined PNG_MODE if defined ANALY_MODE (
    echo ERROR: --png and --png-analy are mutually exclusive
    exit /b 1
)

rem Normalise forward slashes to backslashes
set "PAYLOAD=%PAYLOAD:/=\%"

if not exist "%PAYLOAD%" (
    echo ERROR: payload file not found: %PAYLOAD%
    exit /b 1
)

rem Require payload\ prefix (matches push_run.sh)
if /i not "%PAYLOAD:~0,8%"=="payload\" (
    echo ERROR: payload must be under payload\ ^(got: %PAYLOAD%^)
    exit /b 1
)
set "REL=%PAYLOAD:~8%"

rem Split REL into SUBDIR and FILENAME.  REL is "<sub>\<base>_payload.npz";
rem we take the leaf name and the first path token as subdir (flat layout,
rem one level under payload\, matching the current scripts/ structure).
for %%F in ("%REL%") do set "FILENAME=%%~nxF"
for /f "tokens=1 delims=\" %%S in ("%REL%") do set "SUBDIR=%%S"

set "BASE=%FILENAME:_payload.npz=%"
set "PARAMS=payload\%SUBDIR%\%BASE%_params.json"

rem --- Locate the runner base directory ---
rem Default = this .bat's own folder (%~dp0 includes trailing backslash; strip it).
rem Override via windows_remote.remote_base in hamamatsu_test_config.json if set.
set "SLM_RUNNER=%~dp0"
set "SLM_RUNNER=%SLM_RUNNER:~0,-1%"
if exist hamamatsu_test_config.json (
    for /f "usebackq delims=" %%V in (`python -c "import json; c=json.load(open('hamamatsu_test_config.json')).get('windows_remote',{}); b=c.get('remote_base',''); print(b.replace('/', chr(92))) if b else None"`) do set "SLM_RUNNER=%%V"
)

rem --- Read runner_defaults from params.json, if present ---
set "ETIME="
set "NAVG="
if exist "%PARAMS%" (
    for /f "usebackq tokens=1,2" %%A in (`python -c "import json; d=json.load(open(r'%PARAMS%')).get('runner_defaults',{}); print(d.get('etime_us',''), d.get('n_avg',''))"`) do (
        set "ETIME=%%A"
        set "NAVG=%%B"
    )
)
set "RUNNER_ARGS="
if defined ETIME if not "!ETIME!"=="" set "RUNNER_ARGS=!RUNNER_ARGS! --etime-us !ETIME!"
if defined NAVG  if not "!NAVG!"==""  set "RUNNER_ARGS=!RUNNER_ARGS! --n-avg !NAVG!"

rem --- Stage payload into <runner_base>\incoming\<sub>\ ---
set "INCOMING=%SLM_RUNNER%\incoming\%SUBDIR%"
if not exist "%INCOMING%" mkdir "%INCOMING%"
copy /y "%PAYLOAD%" "%INCOMING%\" >nul
if exist "%PARAMS%" copy /y "%PARAMS%" "%INCOMING%\" >nul
echo [1/3] staged %FILENAME% -^> %INCOMING%\

rem --- Invoke slmrun.bat (same schtasks bridge the SSH flow uses) ---
echo [2/3] slmrun.bat --payload incoming\%SUBDIR%\%FILENAME% --output-prefix %SUBDIR%\%BASE%!RUNNER_ARGS! %HOLD_FLAG%
pushd "%SLM_RUNNER%"
call slmrun.bat --payload incoming\%SUBDIR%\%FILENAME% --output-prefix %SUBDIR%\%BASE%!RUNNER_ARGS! %HOLD_FLAG%
popd

if defined HOLD_FLAG (
    echo [3/3] hold-on: SLM is displaying, no capture, no post-processing.
    echo Done.
    exit /b 0
)

set "DATA_DIR=%SLM_RUNNER%\data\%SUBDIR%"

if defined PNG_MODE (
    echo [3/3] Rendering BMP -^> color-heatmap PNG...
    python "%~dp0render_png.py" "%DATA_DIR%" "%BASE%"
    echo Output:
    echo   %DATA_DIR%\%BASE%_before.png
    echo   %DATA_DIR%\%BASE%_after.png
    echo   %DATA_DIR%\%BASE%_run.json
    exit /b 0
)

if defined ANALY_MODE (
    echo [3/3] Running scripts\sheet\analysis_sheet.py on %BASE%_after.bmp...
    uv run python scripts\sheet\analysis_sheet.py ^
        --after  "%DATA_DIR%\%BASE%_after.bmp" ^
        --before "%DATA_DIR%\%BASE%_before.bmp" ^
        --plot   "%DATA_DIR%\%BASE%_analysis.png" ^
        --result "%DATA_DIR%\%BASE%_analysis.json"
    echo Output:
    echo   %DATA_DIR%\%BASE%_analysis.png
    echo   %DATA_DIR%\%BASE%_analysis.json
    echo   %DATA_DIR%\%BASE%_run.json
    exit /b 0
)

echo [3/3] Captured BMPs in %DATA_DIR%\:
echo   %DATA_DIR%\%BASE%_before.bmp
echo   %DATA_DIR%\%BASE%_after.bmp
echo   %DATA_DIR%\%BASE%_run.json
exit /b 0

:usage
echo Usage: windows_runner\slm_local.bat ^<payload_file^> [--hold-on ^| --png ^| --png-analy]
echo.
echo   ^<payload_file^> must live under payload\ (e.g. payload\sheet\testfile_sheet_payload.npz).
echo   Run from the repo root.  Reads windows_remote.remote_base from
echo   hamamatsu_test_config.json to locate the runner folder (falls back
echo   to this .bat's own directory, i.e. windows_runner\).
exit /b 1
