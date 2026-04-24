@echo off
:: slmrun.bat (lives in the windows_runner\ folder of the SLMengineer repo,
:: canonically C:\Users\Galileo\SLMengineer\windows_runner\).
::
:: Session-1 bridge for runner.py.  Takes the runner.py CLI args,
:: schedules a one-off task via schtasks (which runs in the user's
:: interactive session where the SLM display is accessible), waits
:: for the done.flag, and prints the runner's output.
::
:: Usage (invoked from ssh):
::   cd /d C:\Users\Galileo\SLMengineer\windows_runner
::   slmrun.bat --payload incoming\testfile_lg_payload.npz --output-prefix testfile_lg
::
:: All args after "slmrun.bat" are forwarded to runner.py.

:: DIR = the folder this .bat lives in (location-agnostic).  %~dp0 has a
:: trailing backslash; strip it so path joins like %DIR%\foo stay clean.
set "DIR=%~dp0"
set "DIR=%DIR:~0,-1%"
set "ARGS_FILE=%DIR%\_runner_args.txt"
set "DONE_FLAG=%DIR%\_runner_done.flag"
set "OUT_FILE=%DIR%\_runner_output.txt"
set "TN=SLM_runner"

if "%~1"=="" (
    echo Usage: slmrun.bat [runner.py args]
    exit /b 1
)

:: Clean stale state
del "%DONE_FLAG%" 2>nul
del "%OUT_FILE%" 2>nul

:: Save the full argument list to a file that run_in_session1.bat reads
(echo %*) > "%ARGS_FILE%"

:: Schedule a task that runs run_in_session1.bat in the user's interactive session
schtasks /delete /tn %TN% /f >nul 2>&1
schtasks /create /tn %TN% /tr "\"%DIR%\run_in_session1.bat\"" /sc once /st 00:00 /f >nul 2>&1
schtasks /run /tn %TN% >nul 2>&1

echo [slmrun] Runner scheduled in interactive session, waiting...

:wait
if exist "%DONE_FLAG%" goto done
ping -n 3 127.0.0.1 >nul
goto wait

:done
:: Give run_in_session1.bat a moment to flush stdout
ping -n 2 127.0.0.1 >nul
echo.
if exist "%OUT_FILE%" type "%OUT_FILE%"

:: Cleanup
schtasks /delete /tn %TN% /f >nul 2>&1
del "%DONE_FLAG%" 2>nul
del "%ARGS_FILE%" 2>nul
