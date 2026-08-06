param(
    [string]$BundlePath = $PSScriptRoot
)

$ErrorActionPreference = "Stop"
$bundle = Resolve-Path -LiteralPath $BundlePath -ErrorAction Stop
$required = @(
    "app.py",
    "install-offline.ps1",
    "run-offline.ps1",
    "requirements-offline.txt",
    "wheelhouse",
    "models\whisper-small",
    "SHA256SUMS.csv"
)
foreach ($item in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $bundle.Path $item))) {
        throw "Offline bundle is incomplete: missing $item"
    }
}

$rows = Import-Csv -LiteralPath (Join-Path $bundle.Path "SHA256SUMS.csv")
foreach ($row in $rows) {
    $path = Join-Path $bundle.Path $row.Path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Manifest file is missing: $($row.Path)"
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if ($actual -ne $row.SHA256) {
        throw "Hash verification failed: $($row.Path)"
    }
}
Write-Host "Offline bundle verified: $($rows.Count) files"
