@echo off
setlocal
cd /d "%~dp0"
py setup_app.py --run %*
if errorlevel 1 (
  python setup_app.py --run %*
)
