@echo off
:: Usage: slmrun.bat <script.py>
:: Runs a Python script in the interactive session (Session 1) so the SLM display is accessible.
:: Works from SSH. Waits for completion and prints output.

if "%~1"=="" (
    echo Usage: slmrun.bat yourscript.py
    exit /b 1
)

set "SCRIPT=%~1"
set "BASE=%~n1"
set "DIR=C:\Users\Galileo\SLMengineer"
set "TN=SLM_%BASE%"

del "%DIR%\%BASE%_done.flag" 2>nul
del "%DIR%\%BASE%_output.txt" 2>nul
del "%DIR%\data\%BASE%_output.txt" 2>nul

schtasks /delete /tn %TN% /f >nul 2>&1
schtasks /create /tn %TN% /tr "\"%DIR%\run_in_session1.bat\" %SCRIPT%" /sc once /st 00:00 /f >nul 2>&1
schtasks /run /tn %TN% >nul 2>&1

echo [slmrun] Running %SCRIPT% in interactive session...

:wait
if exist "%DIR%\%BASE%_done.flag" goto done
ping -n 3 127.0.0.1 >nul
goto wait

:done
:: Small delay to let run_in_session1.bat finish moving files
ping -n 2 127.0.0.1 >nul
echo.
:: Output txt is moved to data/ by run_in_session1.bat
if exist "%DIR%\data\%BASE%_output.txt" (
    type "%DIR%\data\%BASE%_output.txt"
) else (
    type "%DIR%\%BASE%_output.txt" 2>nul
)
schtasks /delete /tn %TN% /f >nul 2>&1
del "%DIR%\%BASE%_done.flag" 2>nul
