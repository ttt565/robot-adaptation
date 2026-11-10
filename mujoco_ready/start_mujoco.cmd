@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "ROBOT_PYTHON=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "ROBOT_PYTHON=python"
)
echo Starting the MuJoCo robot simulation.
echo Close the viewer to stop early and save the recorded data.
"%ROBOT_PYTHON%" run_mujoco.py %*
set "ROBOT_EXIT_CODE=%ERRORLEVEL%"
if not "%ROBOT_EXIT_CODE%"=="0" (
    echo.
    echo Simulation failed. Read the error above.
) else (
    echo.
    echo Simulation finished. Output paths are shown above.
)
pause
exit /b %ROBOT_EXIT_CODE%
