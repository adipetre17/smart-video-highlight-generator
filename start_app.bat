@echo off
echo ========================================================
echo   Smart Video Highlight ^& GIF Generator
echo ========================================================
echo.

REM Activate the virtual environment
call "%~dp0venv\Scripts\activate.bat"

echo Virtual environment activated.
echo Starting Streamlit app...
echo.
echo NOTE: If this is your first time running Streamlit on this PC,
echo it might pause and ask for your email address.
echo You can just press ENTER to skip it!
echo.

REM Run Streamlit from the project directory
cd /d "%~dp0"
python -m streamlit run app.py --server.headless=true

pause
