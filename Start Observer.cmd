@echo off
setlocal
cd /d "%~dp0"

rem Select any available Python 3.10+; the Workbench has no third-party runtime
rem dependency and must not assume a particular Conda environment name.
if defined OBSERVER_PYTHON if exist "%OBSERVER_PYTHON%" (
  call :check_python "%OBSERVER_PYTHON%"
  if not errorlevel 1 goto run_observer_python
)

if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
  call :check_python "%CONDA_PREFIX%\python.exe"
  if not errorlevel 1 goto run_conda_prefix
)

where py >nul 2>nul
if not errorlevel 1 (
  py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
  if not errorlevel 1 goto run_py_launcher
)

where python >nul 2>nul
if not errorlevel 1 (
  python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
  if not errorlevel 1 goto run_path_python
)

where conda >nul 2>nul
if not errorlevel 1 (
  conda run -n base python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
  if not errorlevel 1 goto run_conda_base
)

echo Python 3.10 or newer was not found.
echo Activate any suitable Conda environment, set OBSERVER_PYTHON to python.exe,
echo or install Python 3.10+, then run this file again.
exit /b 1

:check_python
"%~1" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
exit /b %errorlevel%

:run_observer_python
"%OBSERVER_PYTHON%" -m observer.launcher
exit /b %errorlevel%

:run_conda_prefix
"%CONDA_PREFIX%\python.exe" -m observer.launcher
exit /b %errorlevel%

:run_py_launcher
py -3 -m observer.launcher
exit /b %errorlevel%

:run_path_python
python -m observer.launcher
exit /b %errorlevel%

:run_conda_base
conda run -n base python -m observer.launcher
exit /b %errorlevel%
