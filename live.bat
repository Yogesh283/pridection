@echo off
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" call "venv\Scripts\activate.bat"
set PYTHONUNBUFFERED=1
echo.
echo Starting LIVE Color + Big/Small prediction (1 hour)...
echo Folder: %cd%
echo Focus: Color + Big/Small
echo.
python -u main.py --live 1
pause
