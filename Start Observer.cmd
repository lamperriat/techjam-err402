@echo off
setlocal
cd /d "%~dp0"

if /I "%CONDA_DEFAULT_ENV%"=="tiktok" if exist "%CONDA_PREFIX%\python.exe" (
  "%CONDA_PREFIX%\python.exe" -m observer.launcher
  exit /b %errorlevel%
)

where conda >nul 2>nul
if errorlevel 1 (
  echo Conda was not found. Activate the tiktok environment, then run this file again.
  exit /b 1
)

conda run -n tiktok python -m observer.launcher
