# Start EXOPHCA with the project's own virtualenv interpreter.
#
# The app needs the versions declared in pyproject.toml: a machine-wide Python
# can carry an older Streamlit whose st.iframe has no alt argument. This
# launcher never guesses: it resolves the interpreter next to the repository
# this file lives in, prints what it is about to use, and fails loudly if the
# virtualenv is missing.
#
#   .\scripts\run_app.ps1
#   .\scripts\run_app.ps1 --server.port 8611

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$entry = Join-Path $projectRoot "app\EXOPHCA.py"

if (-not (Test-Path $python)) {
    Write-Host "No virtualenv interpreter at $python" -ForegroundColor Red
    Write-Host "Create the virtualenv and install pyproject.toml first." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $entry)) {
    Write-Host "No app entry point at $entry" -ForegroundColor Red
    exit 1
}

$version = & $python -c "import streamlit; print(streamlit.__version__)"

Write-Host "Interpreter: $python"
Write-Host "Streamlit:   $version"
Write-Host "Entry point: $entry"
Write-Host ""

Push-Location $projectRoot
try {
    & $python -m streamlit run $entry @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
