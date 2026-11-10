@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "DEMO_PYTHON=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "DEMO_PYTHON=python"
)
"%DEMO_PYTHON%" demo.py %*
set "DEMO_EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %DEMO_EXIT_CODE%
