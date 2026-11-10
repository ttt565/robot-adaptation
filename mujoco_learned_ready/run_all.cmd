@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "LEARNED_PYTHON=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "LEARNED_PYTHON=python"
)
echo Collecting MuJoCo data, training the models, and evaluating predictions.
echo CPU training and camera rendering may take some time.
"%LEARNED_PYTHON%" pipeline.py all %*
set "LEARNED_EXIT_CODE=%ERRORLEVEL%"
if not "%LEARNED_EXIT_CODE%"=="0" (
    echo.
    echo Pipeline failed. Read the error above.
) else (
    echo.
    echo Pipeline finished. Output paths are shown above.
)
pause
exit /b %LEARNED_EXIT_CODE%
