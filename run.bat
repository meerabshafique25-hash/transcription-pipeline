@echo off
REM One-command start for Windows. Shows progress for every step.
cd /d "%~dp0"
echo ==^> [1/3] Setting up Python environment and installing packages...
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo ==^> [2/3] Downloading the Whisper 'small' model (~500 MB, only the first time)...
hf download Systran/faster-whisper-small
echo ==^> [3/3] Starting server. Open http://localhost:8000/docs  (Ctrl+C to stop)
uvicorn main:app --port 8000
