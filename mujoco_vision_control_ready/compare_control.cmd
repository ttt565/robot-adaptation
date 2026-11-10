@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "CONTROL_PYTHON=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "CONTROL_PYTHON=python"
)
echo Comparing three control policies in all cases with seeds 0 and 1.
echo Camera rendering is still required. The experiment may take some time.
"%CONTROL_PYTHON%" run_control.py --compare --all-cases --seeds 0 1 %*
set "CONTROL_EXIT_CODE=%ERRORLEVEL%"
if not "%CONTROL_EXIT_CODE%"=="0" (
    echo.
    echo Comparison failed. Read the error above.
) else (
    echo.
    echo Comparison finished. Output paths are shown above.
)
pause
exit /b %CONTROL_EXIT_CODE%
