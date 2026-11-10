@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "LEARNED_PYTHON=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "LEARNED_PYTHON=python"
)
echo Training models with the existing dataset.
echo CPU training may take some time. Data collection must be completed first.
"%LEARNED_PYTHON%" pipeline.py train %*
set "LEARNED_EXIT_CODE=%ERRORLEVEL%"
if not "%LEARNED_EXIT_CODE%"=="0" (
    echo.
    echo Training failed. Read the error above.
) else (
    echo.
    echo Training finished. Output paths are shown above.
)
pause
exit /b %LEARNED_EXIT_CODE%
