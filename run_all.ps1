# One-command setup + full pipeline on Windows.
#   powershell -ExecutionPolicy Bypass -File run_all.ps1
$ErrorActionPreference = "Stop"
if (-not (Test-Path .venv)) { py -m venv .venv }
.\.venv\Scripts\python -m pip install -q -r requirements.txt
.\.venv\Scripts\python -m pip install -q -e .
.\.venv\Scripts\python -m readmission.eda
.\.venv\Scripts\python -m readmission.train
.\.venv\Scripts\python -m pytest -q
Write-Host "`nDone. Start the API:  .\.venv\Scripts\uvicorn api.main:app"
Write-Host "Start the web app:    .\.venv\Scripts\streamlit run app\streamlit_app.py"
