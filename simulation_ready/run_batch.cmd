@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "SIMULATION_PYTHON=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "SIMULATION_PYTHON=python"
)
echo Running all scenarios with 10 noise seeds. This may take a few minutes.
"%SIMULATION_PYTHON%" run_simulation.py --all-scenarios --seeds 10 --no-animation %*
set "SIMULATION_EXIT_CODE=%ERRORLEVEL%"
if not "%SIMULATION_EXIT_CODE%"=="0" (
    echo.
    echo Batch failed. Read the error above.
) else (
    echo.
    echo Batch finished. Output paths are shown above.
)
pause
exit /b %SIMULATION_EXIT_CODE%
