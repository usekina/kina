param(
    [string]$WhisperModelPath = "$PSScriptRoot\..\models\whisper-small",
    [string]$DatabasePath = "$PSScriptRoot\..\data\kinabot_offline.sqlite3",
    [int]$Port = 8787
)

$ErrorActionPreference = "Stop"
$appRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$python = Join-Path $appRoot.Path ".venv\Scripts\python.exe"
$participantSecret = Join-Path $appRoot.Path ".offline-participant-key"
$apiToken = Join-Path $appRoot.Path ".offline-api-token"
$model = Resolve-Path -LiteralPath $WhisperModelPath -ErrorAction Stop
foreach ($required in @($python, $participantSecret, $apiToken)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Offline installation is incomplete. Run .\install-offline.ps1 first."
    }
}
$env:KINABOT_OFFLINE_RESEARCH_MODE = "true"
$env:KINABOT_OFFLINE_WHISPER_MODEL_PATH = $model.Path
$env:KINABOT_DATABASE_PATH = $DatabasePath
$env:KINABOT_ENVIRONMENT = "offline"
$env:KINABOT_PARTICIPANT_KEY_SECRET = (Get-Content $participantSecret -Raw).Trim()
$env:KINABOT_LOCAL_API_TOKEN = (Get-Content $apiToken -Raw).Trim()
$env:KINABOT_ALLOW_LOCAL_CODES = "false"
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue

Push-Location $appRoot.Path
try {
    & $python -m uvicorn offline_api.api:app --host 127.0.0.1 `
        --port $Port --no-access-log
}
finally {
    Pop-Location
}
