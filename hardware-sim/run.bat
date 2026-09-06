@echo off
title Energy Loss Investigation - Stage 1 telemetry generator
cd /d "%~dp0"

set "PY="
where python >nul 2>&1
if %errorlevel%==0 set "PY=python"
if not "%PY%"=="" goto :found

where py >nul 2>&1
if %errorlevel%==0 set "PY=py"
if not "%PY%"=="" goto :found

if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not "%PY%"=="" goto :found

if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe" set "PY=%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe"
if not "%PY%"=="" goto :found

if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe" set "PY=%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe"
if not "%PY%"=="" goto :found

if exist "%USERPROFILE%\AppData\Local\Microsoft\WindowsApps\python.exe" set "PY=%USERPROFILE%\AppData\Local\Microsoft\WindowsApps\python.exe"
if not "%PY%"=="" goto :found

echo.
echo   Could not find Python on this machine.
echo   Open Anaconda Prompt in this folder and run:
echo       python simulate.py --ticks 300 --out telemetry.ndjson
echo       python validate.py telemetry.ndjson
goto :end

:found
echo.
echo   Python: %PY%
echo   Folder: %CD%
echo.
echo   ================================================================
echo    GENERATING TELEMETRY  -  4 modes x 300 simulated minutes
echo   ================================================================
"%PY%" simulate.py --ticks 300 --out telemetry.ndjson
if errorlevel 1 goto :end

echo.
echo   ================================================================
echo    VALIDATING
echo   ================================================================
"%PY%" validate.py telemetry.ndjson

echo.
echo   Output file: %CD%\telemetry.ndjson

:end
echo.
pause
