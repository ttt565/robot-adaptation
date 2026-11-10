@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "SIMULATION_PYTHON=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "SIMULATION_PYTHON=python"
)
echo Starting robot simulation. Close the animation window when finished.
"%SIMULATION_PYTHON%" run_simulation.py %*
set "SIMULATION_EXIT_CODE=%ERRORLEVEL%"
if not "%SIMULATION_EXIT_CODE%"=="0" (
    echo.
    echo Simulation failed. Read the error above.
) else (
    echo.
    echo Simulation finished. Output paths are shown above.
)
pause
exit /b %SIMULATION_EXIT_CODE%
