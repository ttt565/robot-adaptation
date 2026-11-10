@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "VISION_PYTHON=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "VISION_PYTHON=python"
)
echo Running all camera and dynamics cases without the interactive viewer.
"%VISION_PYTHON%" run_vision.py --all-cases --headless %*
set "VISION_EXIT_CODE=%ERRORLEVEL%"
if not "%VISION_EXIT_CODE%"=="0" (
    echo.
    echo Checks failed. Read the error above.
) else (
    echo.
    echo Checks finished. Output paths are shown above.
)
pause
exit /b %VISION_EXIT_CODE%
