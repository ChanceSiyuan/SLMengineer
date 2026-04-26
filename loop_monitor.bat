@echo off
title SLM Loop Monitor
:loop
cls
echo === %date% %time% ===
echo.
echo --- python.exe processes ---
tasklist /FI "IMAGENAME eq python.exe" /FO TABLE
echo.
echo --- newest iter PNGs ---
for /f "delims=" %%f in ('dir /O-D /B /S "%USERPROFILE%\SLMengineer\data\sheet_inloop_*\iter_*.png" 2^>nul') do @echo %%f
echo.
echo --- last after-BMP timestamp ---
dir "%USERPROFILE%\SLMengineer\data\sheet\testfile_sheet_after.bmp" 2>nul | findstr testfile_sheet_after
echo.
echo (refresh in 5s, Ctrl+C to close)
timeout /t 5 /nobreak >nul
goto loop
