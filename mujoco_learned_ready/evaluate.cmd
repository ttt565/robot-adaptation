@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "LEARNED_PYTHON=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "LEARNED_PYTHON=python"
)
echo Evaluating the existing trained models on the recorded dataset.
"%LEARNED_PYTHON%" pipeline.py evaluate %*
set "LEARNED_EXIT_CODE=%ERRORLEVEL%"
if not "%LEARNED_EXIT_CODE%"=="0" (
    echo.
    echo Evaluation failed. Read the error above.
) else (
    echo.
    echo Evaluation finished. Output paths are shown above.
)
pause
exit /b %LEARNED_EXIT_CODE%
