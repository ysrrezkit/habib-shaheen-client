@echo off
echo Checking for required libraries...
pip install -r requirements.txt

echo Opening website...
start http://127.0.0.1:8050

echo Starting the dashboard...
python app.py

pause