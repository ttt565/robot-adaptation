@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "ROBOT_PYTHON=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "ROBOT_PYTHON=python"
)
echo Comparing six controllers in MuJoCo without opening the viewer.
"%ROBOT_PYTHON%" run_mujoco.py --compare --headless %*
set "ROBOT_EXIT_CODE=%ERRORLEVEL%"
if not "%ROBOT_EXIT_CODE%"=="0" (
    echo.
    echo Comparison failed. Read the error above.
) else (
    echo.
    echo Comparison finished. Output paths are shown above.
)
pause
exit /b %ROBOT_EXIT_CODE%
