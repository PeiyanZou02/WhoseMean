@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" diffusion_pipeline.py prepare --classification Painting
if errorlevel 1 goto error
".venv\Scripts\python.exe" diffusion_pipeline.py train-lora --classification Painting --steps 1200
if errorlevel 1 goto error
".venv\Scripts\python.exe" diffusion_pipeline.py generate --classification Painting --strength 0.62 --steps 35 --seed 7
if errorlevel 1 goto error
echo.
echo Coherent mean painting is ready in outputs\diffusion\painting.
pause
exit /b 0
:error
echo.
echo Coherent mean pipeline failed. See the message above.
pause
exit /b 1
