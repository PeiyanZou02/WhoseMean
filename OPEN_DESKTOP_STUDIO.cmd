@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Python environment not found.
  pause
  exit /b 1
)
start "Whose Mean Desktop" ".venv\Scripts\pythonw.exe" "desktop_studio.py"
