@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" nga_pipeline.py collect --limit 300
if errorlevel 1 goto error
".venv\Scripts\python.exe" nga_pipeline.py mean
if errorlevel 1 goto error
".venv\Scripts\python.exe" nga_pipeline.py train-hd --epochs 50
if errorlevel 1 goto error
echo.
echo NGA HD mean image is ready. Refresh http://127.0.0.1:8765/
pause
exit /b 0
:error
echo.
echo NGA pipeline failed. See the message above.
pause
exit /b 1
