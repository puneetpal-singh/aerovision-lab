@echo off
cd /d "%~dp0"
if not exist "models\aerovision-full-fp32.onnx" (
    echo Model missing. Follow the model setup instructions in README.md.
    pause
    exit /b 1
)
if not exist ".demo-venv\Scripts\python.exe" (
    py -3.12 -m venv .demo-venv
    if errorlevel 1 (
        echo Python 3.12 is required. Install it and try again.
        pause
        exit /b 1
    )
)
".demo-venv\Scripts\python.exe" -m pip install -r requirements-demo.txt
if errorlevel 1 (
    pause
    exit /b 1
)
".demo-venv\Scripts\python.exe" -m streamlit run demo.py
pause
