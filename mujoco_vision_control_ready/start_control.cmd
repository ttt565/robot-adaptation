@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "CONTROL_PYTHON=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "CONTROL_PYTHON=python"
)
echo Starting the MuJoCo vision and motion control experiment.
"%CONTROL_PYTHON%" run_control.py %*
set "CONTROL_EXIT_CODE=%ERRORLEVEL%"
if not "%CONTROL_EXIT_CODE%"=="0" (
    echo.
    echo Experiment failed. Read the error above.
) else (
    echo.
    echo Experiment finished. Output paths are shown above.
)
pause
exit /b %CONTROL_EXIT_CODE%
