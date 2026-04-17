@echo off
:: run_in_session1.bat (inside C:\Users\Galileo\slm_runner\)
::
:: Invoked by the schtasks task created in slmrun.bat.  Runs in the
:: user's interactive session, where the SLM display is accessible.
::
:: Reads runner.py CLI args from _runner_args.txt, runs runner.py with
:: them, writes stdout/stderr to _runner_output.txt, and creates
:: _runner_done.flag to signal completion back to slmrun.bat.

set "DIR=C:\Users\Galileo\slm_runner"
set "MAIN_PYTHON=C:\Users\Galileo\SLMengineer\.venv\Scripts\python.exe"
set "ARGS_FILE=%DIR%\_runner_args.txt"
set "OUT_FILE=%DIR%\_runner_output.txt"
set "DONE_FLAG=%DIR%\_runner_done.flag"

cd /d "%DIR%"

:: Read the arg line saved by slmrun.bat
set /p RUNNER_ARGS=<"%ARGS_FILE%"

:: Kill stale python processes that may be holding display or camera locks
taskkill /f /im python.exe >nul 2>&1

:: Run the runner with the saved args.  Note: PYTHONPATH gets the
:: main SLMengineer repo so runner.py can import slm.display and
:: slm.camera (the runner adds C:\Users\Galileo\SLMengineer\src to
:: sys.path internally as well, this is a belt-and-braces).
set "PYTHONPATH=C:\Users\Galileo\SLMengineer\src"
"%MAIN_PYTHON%" runner.py %RUNNER_ARGS% > "%OUT_FILE%" 2>&1

:: Signal completion
echo done > "%DONE_FLAG%"
