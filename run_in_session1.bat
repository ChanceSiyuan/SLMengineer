@echo off
cd /d "C:\Users\Galileo\SLMengineer"
set PYTHONPATH=src
if not exist data mkdir data
:: Kill stale python processes from previous runs to release file locks
taskkill /f /im python.exe >nul 2>&1
:: Clean cached bytecode to ensure fresh imports after sync
for /d /r src %%d in (__pycache__) do if exist "%%d" rd /s /q "%%d"
.venv\Scripts\python.exe %1 > "%~n1_output.txt" 2>&1
:: Move generated data/image/json/txt files into data/ to keep repo root clean
for %%f in (*.npy *.png *_run.json) do move /Y "%%f" data\ >nul 2>&1
:: Signal completion, then move output txt into data/ too
echo done > "%~n1_done.flag"
move /Y "%~n1_output.txt" data\ >nul 2>&1
