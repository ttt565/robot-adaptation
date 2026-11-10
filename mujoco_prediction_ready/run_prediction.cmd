@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "PREDICTION_PYTHON=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "PREDICTION_PYTHON=python"
)
echo Collecting MuJoCo data and replaying four predictors on the same logs.
echo Camera rendering is required. This experiment may take some time.
"%PREDICTION_PYTHON%" run_prediction.py %*
set "PREDICTION_EXIT_CODE=%ERRORLEVEL%"
if not "%PREDICTION_EXIT_CODE%"=="0" (
    echo.
    echo Experiment failed. Read the error above.
) else (
    echo.
    echo Experiment finished. Output paths are shown above.
)
pause
exit /b %PREDICTION_EXIT_CODE%
