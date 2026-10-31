@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating environment...
call venv\Scripts\activate

echo Installing requirements...
pip install -r requirements.txt

echo Starting Flask server...
python app.py

pause
