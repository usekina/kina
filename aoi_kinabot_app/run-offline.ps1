param(
    [string]$WhisperModelPath = "$PSScriptRoot\models\whisper-small",
    [string]$DatabasePath = "$PSScriptRoot\data\kinabot_offline.sqlite3",
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$resolvedModel = Resolve-Path -LiteralPath $WhisperModelPath -ErrorAction Stop
$offlinePython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$secretPath = Join-Path $PSScriptRoot ".offline-participant-key"
if (-not (Test-Path -LiteralPath $offlinePython -PathType Leaf)) {
    throw "Offline environment not installed. Run .\install-offline.ps1 first."
}
if (-not (Test-Path -LiteralPath $secretPath -PathType Leaf)) {
    throw "Offline participant-key secret is missing. Run .\install-offline.ps1."
}
$env:KINABOT_OFFLINE_RESEARCH_MODE = "true"
$env:KINABOT_OFFLINE_WHISPER_MODEL_PATH = $resolvedModel.Path
$env:KINABOT_DATABASE_PATH = $DatabasePath
$env:KINABOT_ENVIRONMENT = "offline"
$env:KINABOT_PARTICIPANT_KEY_SECRET = (
    Get-Content -LiteralPath $secretPath -Raw
).Trim()
$env:KINABOT_ALLOW_LOCAL_CODES = "false"
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue

& $offlinePython -m streamlit run "$PSScriptRoot\app.py" `
    --server.address=127.0.0.1 `
    --server.port=$Port `
    --server.headless=true `
    --browser.gatherUsageStats=false
