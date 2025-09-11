@echo off
echo Face Recognition Identity Retrieval
echo ===================================
echo.

echo Starting face recognition system...
python live_face_recognition.py

echo.
echo Reading confirmed identity...

if exist recognition_result.txt (
    set /p identity=<recognition_result.txt
    
    if "!identity!"=="NONE" (
        echo No identity confirmed.
        echo RESULT: NONE
    ) else (
        echo Identity confirmed: !identity!
        echo RESULT: !identity!
        
        REM Example usage - you can modify this section
        echo.
        echo You can now use the confirmed identity for:
        echo - Access control for !identity!
        echo - Logging attendance for !identity!
        echo - Personalized services for !identity!
    )
) else (
    echo Error: Result file not found.
    echo RESULT: ERROR
)

echo.
pause