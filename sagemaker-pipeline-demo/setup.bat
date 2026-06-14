@echo off
echo Creating virtual environment...
python -m venv .venv

echo.
echo Activating virtual environment...
call .venv\Scripts\activate

echo.
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo Setup complete!
echo.
echo To activate the environment in the future:
echo   .venv\Scripts\activate
echo.
echo To run the pipeline:
echo   python run_local.py
echo ========================================
