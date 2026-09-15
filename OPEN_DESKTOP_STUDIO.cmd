@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Python environment not found.
  pause
  exit /b 1
)
set "PYTHONPATH=%~dp0src"
start "Whose Mean Desktop" ".venv\Scripts\pythonw.exe" -m whose_mean.app
