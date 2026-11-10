@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "VISION_PYTHON=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "VISION_PYTHON=python"
)
echo Starting the MuJoCo robot camera experiment.
"%VISION_PYTHON%" run_vision.py %*
set "VISION_EXIT_CODE=%ERRORLEVEL%"
if not "%VISION_EXIT_CODE%"=="0" (
    echo.
    echo Experiment failed. Read the error above.
) else (
    echo.
    echo Experiment finished. Output paths are shown above.
)
pause
exit /b %VISION_EXIT_CODE%
