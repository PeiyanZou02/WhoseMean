@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" nga_pipeline.py visualize-data
if errorlevel 1 goto error
".venv\Scripts\python.exe" nga_pipeline.py train-512 --epochs 50 --batch-size 1
if errorlevel 1 goto error
echo.
echo NGA 512 model and interactive training frames are ready.
echo Refresh http://127.0.0.1:8765/
pause
exit /b 0
:error
echo.
echo NGA 512 pipeline failed. See the message above.
pause
exit /b 1
