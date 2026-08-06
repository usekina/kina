param(
    [string]$PythonCommand = "python"
)

$ErrorActionPreference = "Stop"
$appRoot = $PSScriptRoot
$wheelhouse = Join-Path $appRoot "wheelhouse"
$modelPath = Join-Path $appRoot "models\whisper-small"
$venvPython = Join-Path $appRoot ".venv\Scripts\python.exe"
$secretPath = Join-Path $appRoot ".offline-participant-key"

if (-not (Test-Path -LiteralPath $wheelhouse -PathType Container)) {
    throw "Missing wheelhouse. Ask the package administrator for the complete offline bundle."
}
if (-not (Test-Path -LiteralPath $modelPath -PathType Container)) {
    throw "Missing local Whisper model at models\whisper-small."
}

& $PythonCommand -m venv (Join-Path $appRoot ".venv")
& $venvPython -m pip install --no-index --find-links $wheelhouse `
    -r (Join-Path $appRoot "requirements-offline.txt")
& $venvPython -m pytest --version 2>$null

if (-not (Test-Path -LiteralPath $secretPath -PathType Leaf)) {
    $bytes = New-Object byte[] 48
    [System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
    [System.IO.File]::WriteAllText(
        $secretPath,
        [Convert]::ToBase64String($bytes),
        [System.Text.UTF8Encoding]::new($false)
    )
}

Write-Host "KinaBot offline installation completed."
Write-Host "Run .\run-offline.ps1 to start."
