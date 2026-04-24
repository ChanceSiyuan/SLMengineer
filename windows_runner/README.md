# Windows hardware runner

A lightweight, dedicated folder on the Windows lab box that does nothing
except load precomputed SLM phase payloads, display them on the SLM,
capture the camera, and return the data to Linux.

**All CGM/WGS/Fresnel/calibration compute stays on the Linux side.** The
Windows runner is a pure hardware-I/O layer.

## Intended layout (on Windows)

The runner lives **inside the SLMengineer repo checkout** — there is no
separate standalone directory to maintain. After `git pull` on the
Windows box, the tree looks like:

```
C:\Users\Galileo\SLMengineer\
├── src\                  (Linux-shared source, imported by runner.py)
├── .venv\                (Python env; numpy/matplotlib/Pillow + SLM/camera deps)
├── hamamatsu_test_config.json
└── windows_runner\
    ├── runner.py         (hardware entry point)
    ├── slmrun.bat        (SSH-invoked Session-1 bridge)
    ├── run_in_session1.bat
    ├── slm_display.py
    ├── vimba_camera.py
    ├── holdon.py / takeshot.py / render_png.py / gaussian_fit.py
    ├── incoming\         (Linux scp's payload .npz files here) — gitignored
    ├── data\             (runner writes before/after captures here) — gitignored
    └── _runner_{args,done,output}.*   (Session-0/1 IPC; gitignored)
```

The Linux-side orchestrator (`push_run.sh` at the SLMengineer repo
root) is the universal entry point. Given a local
`payload/<sub>/<base>_payload.npz`, it:

1. `scp`'s the payload (+ sibling params.json) into
   `C:\Users\Galileo\SLMengineer\windows_runner\incoming\<sub>\`,
2. `ssh`'s into the Windows box and runs `slmrun.bat --payload
   incoming\<sub>\<base>_payload.npz --output-prefix <sub>\<base>`,
3. `scp`'s the captured `data\<sub>\<base>_{before,after}.bmp` +
   `_run.json` back to `./data/<sub>/` on Linux.

The CGM/WGS compute itself runs earlier, locally, via
`uv run python scripts/<sub>/testfile_<sub>.py`.

## One-time Windows setup

### 1. Clone / sync the SLMengineer repo

```cmd
cd C:\Users\Galileo
git clone <repo-url> SLMengineer
```

(Or `git pull` if it is already there.) That single clone gives you the
runner — no separate copy step.

### 2. Create `incoming\` and `data\` (first run only)

`push_run.sh` creates these automatically via SSH, but if you want to
run locally (`slm_local.bat`) before the first SSH run:

```cmd
mkdir C:\Users\Galileo\SLMengineer\windows_runner\incoming
mkdir C:\Users\Galileo\SLMengineer\windows_runner\data
```

### 3. Create the Python virtual environment

The runner imports `slm.display.SLMdisplay` and `slm.camera.VimbaCamera`
from the repo's `src/slm/` package (the runner adds that path to
`sys.path` automatically). So you need a venv with:

- numpy
- matplotlib
- Pillow
- slmpy (or whatever `slm.display` actually depends on)
- The Allied Vision Vimba SDK Python bindings (for `slm.camera.VimbaCamera`)

```cmd
cd C:\Users\Galileo\SLMengineer
python -m venv .venv
.venv\Scripts\activate
pip install numpy matplotlib Pillow slmpy
:: Install the Vimba SDK Python bindings per Allied Vision's docs
:: (typically a .whl download from their site)
pip install <path-to-vmbpy-wheel>
deactivate
```

`run_in_session1.bat` invokes the venv at
`C:\Users\Galileo\SLMengineer\.venv\Scripts\python.exe` explicitly, so
one repo-level venv covers everything.

### 4. Smoke test

```cmd
cd C:\Users\Galileo\SLMengineer\windows_runner
..\.venv\Scripts\python.exe runner.py --help
```

You should see the argparse help text listing `--payload`,
`--output-prefix`, `--etime-us`, `--n-avg`, `--monitor`. If you get
`ModuleNotFoundError: slm.display`, the `src/` layout has changed —
check `_MAIN_REPO_SRC` at the top of `runner.py`.

## Per-experiment workflow (from Linux)

```bash
# On the Linux dev box, from the SLMengineer repo root:
uv run python scripts/lg/testfile_lg.py            # local CGM compute
./push_run.sh payload/lg/testfile_lg_payload.npz   # push / run / pull
```

`push_run.sh` covers:

- **[1/4]** ensure `incoming\<sub>\` exists on Windows
- **[2/4]** scp the payload .npz + params.json into `incoming\<sub>\`
- **[3/4]** ssh trigger `slmrun.bat` → `runner.py`
- **[4/4]** scp `data\<sub>\<base>_{before,after}.bmp` + `_run.json`
  back into `./data/<sub>/`

Add `--hold-on` to display the payload on the SLM indefinitely (no
capture, no pull). Expected wall time per experiment: **~2 minutes**
(CGM dominates; hardware I/O and file transfer are < 10 s combined
over gigabit).

## Writing a new experiment

The Windows runner is target-agnostic. To run a different experiment
(e.g. a new target or a different optimisation algorithm), you only
need to add one file on the **Linux** side:

1. Copy `scripts/lg/testfile_lg.py` to
   `scripts/<shape>/testfile_<shape>.py`, update `OUTPUT_DIR =
   "payload/<shape>"`, and change the target / CGM config as needed.
2. Run it, then push/run via
   `./push_run.sh payload/<shape>/testfile_<shape>_payload.npz`.

The Windows runner (`runner.py`) and the universal `push_run.sh` stay
unchanged across all experiments as long as the payload format stays
the same (`.npz` with a `slm_screen` uint8 key of shape `(SLM_H, SLM_W)`).

## Troubleshooting

- **"payload not found"**: the scp step failed; check the SSH
  connection and that `incoming\` exists on the Windows box.
- **SLM black or unresponsive**: verify the SLM is on monitor 1
  (`--monitor 1` default) and is powered on.
- **Camera timeout**: the Vimba camera may need to be power-cycled;
  also ensure no other process is holding it open.
- **`ModuleNotFoundError: slm.display`**: the runner couldn't find
  `src/slm/` under the repo root. Edit `_MAIN_REPO_SRC` near the top
  of `runner.py`.
- **Calibration correction looks wrong**: the correction is applied on
  the **Linux** side by `scripts/<shape>/testfile_<shape>.py` via
  `slm.imgpy.SLM_screen_Correct` using
  `calibration/CAL_LSH0905549_1013nm.bmp`. If you need to change the
  calibration file, edit the Linux script — the runner has no
  calibration logic at all.

## Migrating from the old `C:\Users\Galileo\slm_runner\` layout

Earlier versions kept the runtime workspace in a sibling folder
`C:\Users\Galileo\slm_runner\`, hand-maintained by copying files out of
this repo. The two have been merged — everything now lives inside
`windows_runner\` in the repo. One-time migration on the Windows box:

```cmd
cd C:\Users\Galileo\SLMengineer && git pull
robocopy C:\Users\Galileo\slm_runner\incoming C:\Users\Galileo\SLMengineer\windows_runner\incoming /E
robocopy C:\Users\Galileo\slm_runner\data     C:\Users\Galileo\SLMengineer\windows_runner\data /E
:: Confirm runs are working from the new location, then remove the old folder:
rmdir /s /q C:\Users\Galileo\slm_runner
```
