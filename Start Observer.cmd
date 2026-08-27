@echo off
setlocal
cd /d "%~dp0"

if exist "D:\450\conda\envs\tiktok\python.exe" (
  "D:\450\conda\envs\tiktok\python.exe" -m observer.launcher
) else (
  conda run -n tiktok python -m observer.launcher
)
