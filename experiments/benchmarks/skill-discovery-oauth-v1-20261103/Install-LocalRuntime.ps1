[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$runtimeRoot = Join-Path $PSScriptRoot 'tools\llama-b11960-cuda12.4'
if (Test-Path -LiteralPath $runtimeRoot) { throw 'Target exists. Preserve it; use the existing runtime or a new path.' }
$downloadRoot = Join-Path $PSScriptRoot 'downloads'
New-Item -ItemType Directory -Path $downloadRoot -Force | Out-Null
$archives = @(
    @{ name = 'llama-b11960-bin-win-cuda-12.4-x64.zip'; sha256 = '8d9d819261de10cc2f03921b8b79c779c5c458e21edb60c170ee30a7d462d684' },
    @{ name = 'cudart-llama-bin-win-cuda-12.4-x64.zip'; sha256 = '8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6' }
)
foreach ($archive in $archives) {
    $archivePath = Join-Path $downloadRoot $archive.name
    $url = 'https://github.com/ggml-org/llama.cpp/releases/download/b11960/' + $archive.name
    if (-not (Test-Path -LiteralPath $archivePath)) { Invoke-WebRequest -Uri $url -OutFile ($archivePath + '.part'); Move-Item -LiteralPath ($archivePath + '.part') -Destination $archivePath }
    $actual = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $archive.sha256) { throw "Checksum mismatch for $($archive.name). Download preserved for inspection." }
}
New-Item -ItemType Directory -Path $runtimeRoot | Out-Null
foreach ($archive in $archives) {
    # Archives come from the official release and are hash-pinned; extract within a fresh package-local directory.
    Expand-Archive -LiteralPath (Join-Path $downloadRoot $archive.name) -DestinationPath $runtimeRoot -Force
}
$server = Get-ChildItem -LiteralPath $runtimeRoot -Filter llama-server.exe -Recurse
if ($server.Count -ne 1) { throw 'Could not identify exactly one llama-server.exe.' }
Write-Host "CUDA runtime installed locally: $($server.FullName)"
Write-Host 'No model loaded or inference performed. No PATH/global install changed.'
