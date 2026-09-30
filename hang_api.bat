@echo off
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" call "venv\Scripts\activate.bat"
set PYTHONUNBUFFERED=1
echo.
echo HANG on WinGo history API (fast poll)...
echo Ctrl+C to stop.
echo.
python -u tools\hang_history_api.py --interval 0.4 --compare-dear
pause
