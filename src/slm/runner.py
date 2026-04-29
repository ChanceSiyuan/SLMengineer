"""Cross-platform wrapper around ``push_run.{ps1,sh}`` for closed-loop SLM
experiments.  Streams the script's stdout into the calling Python process,
enforces a wall-clock timeout, and retries on transient failures (e.g. SSH
drops) without losing the CGM compute that produced the payload.
"""
from __future__ import annotations

import platform
import subprocess
import sys
import time
from pathlib import Path


def push_run_retry(
    payload_arg: str,
    *,
    repo_root: str | Path,
    etime_us: int = 1500,
    n_avg: int = 20,
    hold_on: bool = False,
    timeout_sec: float | None = 180,
    retry_count: int = 3,
    retry_sleep: float = 10,
    stream: bool = True,
) -> None:
    """Run ``push_run.{ps1,sh} <payload_arg>`` with streaming + timeout + retry.

    Parameters
    ----------
    payload_arg : str
        Path to the payload, **relative to ``repo_root``** and rooted at
        ``payload/`` (e.g. ``"payload/sheet/foo_payload.npz"``).  This is
        what the underlying scripts expect as their first positional arg.
    repo_root : str or Path
        Repository root (where ``push_run.ps1`` / ``push_run.sh`` live and
        where the script must be ``cd``'d into for relative paths).
    etime_us : int
        Camera exposure time in microseconds; forwarded as
        ``-EtimeUs`` (Windows) or ``--etime-us`` (Linux).
    n_avg : int
        Number of frames averaged per capture; forwarded as
        ``-NAvg`` / ``--n-avg``.
    hold_on : bool
        Display payload on the SLM and hold without capturing/pulling.
    timeout_sec : float or None
        Wall-clock timeout per attempt; ``None`` disables.
    retry_count : int
        Total attempts before giving up (including the first try).
    retry_sleep : float
        Seconds to sleep between attempts.
    stream : bool
        If True, pipe child stdout/stderr into the parent process line by
        line (matches the legacy notebook behaviour).  If False, just
        ``subprocess.run`` and let the child inherit stdio.
    """
    repo_root = Path(repo_root)
    is_windows = platform.system() == "Windows"
    script = repo_root / ("push_run.ps1" if is_windows else "push_run.sh")

    if is_windows:
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy", "Bypass",
            "-File", str(script),
            payload_arg,
            "-EtimeUs", str(etime_us),
            "-NAvg", str(n_avg),
        ]
        if hold_on:
            cmd.append("-HoldOn")
    else:
        cmd = [
            "bash", str(script),
            payload_arg,
            "--etime-us", str(etime_us),
            "--n-avg", str(n_avg),
        ]
        if hold_on:
            cmd.append("--hold-on")

    for attempt in range(1, retry_count + 1):
        try:
            if stream:
                p = subprocess.Popen(
                    cmd,
                    cwd=str(repo_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                try:
                    assert p.stdout is not None
                    for line in p.stdout:
                        sys.stdout.write(line)
                    ret = p.wait(timeout=timeout_sec)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                    raise RuntimeError(
                        f"push_run timed out after {timeout_sec}s"
                    )
                if ret != 0:
                    raise RuntimeError(
                        f"push_run exited with code {ret}"
                    )
            else:
                subprocess.run(
                    cmd, cwd=str(repo_root),
                    check=True, timeout=timeout_sec,
                )
            return
        except Exception as e:
            if attempt == retry_count:
                raise
            print(
                f"[RETRY] push_run attempt {attempt}/{retry_count} "
                f"failed: {e}; sleeping {retry_sleep}s",
                file=sys.stderr,
            )
            time.sleep(retry_sleep)
