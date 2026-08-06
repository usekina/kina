param(
    [Parameter(Mandatory = $true)]
    [string]$WheelhousePath,
    [Parameter(Mandatory = $true)]
    [string]$WhisperModelPath,
    [string]$OutputDirectory = "$PSScriptRoot\dist"
)

$ErrorActionPreference = "Stop"
$wheelhouse = Resolve-Path -LiteralPath $WheelhousePath -ErrorAction Stop
$model = Resolve-Path -LiteralPath $WhisperModelPath -ErrorAction Stop
$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$commit = (& git -C $repoRoot.Path rev-parse --short=12 HEAD).Trim()
if (-not $commit) { throw "Unable to determine the Git commit." }

$output = New-Item -ItemType Directory -Force -Path $OutputDirectory
$stage = Join-Path $output.FullName "KinaBot-Offline-$commit"
$archive = Join-Path $output.FullName "source-$commit.zip"
if (Test-Path -LiteralPath $stage) {
    throw "Bundle staging directory already exists: $stage"
}

& git -C $repoRoot.Path archive --format=zip --output=$archive HEAD aoi_kinabot_app
Expand-Archive -LiteralPath $archive -DestinationPath $output.FullName
Move-Item -LiteralPath (Join-Path $output.FullName "aoi_kinabot_app") -Destination $stage
Remove-Item -LiteralPath $archive

Copy-Item -LiteralPath $wheelhouse.Path -Destination (Join-Path $stage "wheelhouse") -Recurse
New-Item -ItemType Directory -Force -Path (Join-Path $stage "models") | Out-Null
Copy-Item -LiteralPath $model.Path -Destination (Join-Path $stage "models\whisper-small") -Recurse

$manifestPath = Join-Path $stage "SHA256SUMS.csv"
$rows = Get-ChildItem -LiteralPath $stage -Recurse -File | ForEach-Object {
    $relative = $_.FullName.Substring($stage.Length + 1)
    $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
    [pscustomobject]@{ Path = $relative; SHA256 = $hash.Hash; Bytes = $_.Length }
}
$rows | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding utf8

$zipPath = "$stage.zip"
Compress-Archive -LiteralPath $stage -DestinationPath $zipPath
Write-Host "Created $zipPath"
Write-Host "Commit $commit; files $($rows.Count)"
