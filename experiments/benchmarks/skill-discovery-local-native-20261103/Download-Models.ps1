[CmdletBinding()]
param([ValidateSet('0.6B','4B','Both')][string]$Model = '0.6B')
$ErrorActionPreference = 'Stop'
$modelDir = Join-Path $PSScriptRoot 'models'
New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
$sources = @(
    @{ size='0.6B'; repo='Qwen/Qwen3-0.6B-GGUF'; revision='23749fefcc72300e3a2ad315e1317431b06b590a'; file='Qwen3-0.6B-Q8_0.gguf'; bytes=639446688; sha256='9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031' },
    @{ size='4B'; repo='Qwen/Qwen3-4B-GGUF'; revision='bc640142c66e1fdd12af0bd68f40445458f3869b'; file='Qwen3-4B-Q4_K_M.gguf'; bytes=2497280256; sha256='7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5' }
)
foreach ($source in $sources) {
    if ($Model -ne 'Both' -and $source.size -ne $Model) { continue }
    $target = Join-Path $modelDir $source.file
    if (-not (Test-Path -LiteralPath $target)) {
        $url = 'https://huggingface.co/' + $source.repo + '/resolve/' + $source.revision + '/' + $source.file
        Invoke-WebRequest -Uri $url -OutFile ($target + '.part')
        if ((Get-Item -LiteralPath ($target + '.part')).Length -ne $source.bytes) { throw 'Downloaded model size mismatch. Partial file preserved.' }
        $actual = (Get-FileHash -LiteralPath ($target + '.part') -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $source.sha256) { throw 'Model checksum mismatch. Partial file preserved.' }
        Move-Item -LiteralPath ($target + '.part') -Destination $target
    }
    if ((Get-Item -LiteralPath $target).Length -ne $source.bytes -or (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $source.sha256) {
        throw 'Existing model differs from the pinned official file. It was not replaced.'
    }
    $source | ConvertTo-Json | Set-Content -LiteralPath ($target + '.source.json') -Encoding utf8
    Write-Host "Verified: $($source.file), $($source.bytes) bytes"
}
Write-Host 'Download only: no server started and no inference performed.'
