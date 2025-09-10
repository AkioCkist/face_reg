@echo off
echo Face Recognition System
echo ---------------------
echo.

echo Step 1: Creating face database...
python face_db.py

echo.
echo Step 2: Starting live face recognition...
python live_face_recognition.py

echo.
echo Face recognition completed!
pause
