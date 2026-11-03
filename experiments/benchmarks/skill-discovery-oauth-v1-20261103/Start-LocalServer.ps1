[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ServerExe,
    [Parameter(Mandatory)][string]$ModelFile,
    [string]$Config = (Join-Path $PSScriptRoot 'configs\local-qwen3-06b.json')
)
$ErrorActionPreference = 'Stop'
$serverPath = (Resolve-Path -LiteralPath $ServerExe).Path
$modelPath = (Resolve-Path -LiteralPath $ModelFile).Path
$configPath = (Resolve-Path -LiteralPath $Config).Path
$cfg = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if ($cfg.backend -ne 'llama') { throw 'Config must use backend llama.' }
if ($cfg.model -notmatch '^[A-Za-z0-9._-]+$') { throw 'Use a simple model alias.' }
$endpoint = [Uri]$cfg.base_url
if ($endpoint.Host -ne '127.0.0.1' -or $endpoint.Scheme -ne 'http') { throw 'Server must bind to 127.0.0.1.' }
$runtimeDir = Join-Path $PSScriptRoot ('runtime\' + $cfg.model)
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$proofPath = Join-Path $runtimeDir 'gpu-proof.json'
$previousProof = $null
if (Test-Path -LiteralPath $proofPath) {
    $previousProof = Get-Content -LiteralPath $proofPath -Raw | ConvertFrom-Json
    $previousProcess = Get-Process -Id $previousProof.pid -ErrorAction SilentlyContinue
    if ($previousProcess -and $previousProcess.StartTime.ToUniversalTime().ToString('o') -eq $previousProof.process_start) {
        throw "The recorded server PID $($previousProof.pid) is still running. Stop that exact process before starting another."
    }
}
$launchDir = Join-Path $runtimeDir ('launch-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfff') + '-' + [guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Path $launchDir | Out-Null
if ($previousProof) { Copy-Item -LiteralPath $proofPath -Destination (Join-Path $launchDir 'previous-gpu-proof.json') }
$stdoutPath = Join-Path $launchDir 'server.stdout.log'
$stderrPath = Join-Path $launchDir 'server.stderr.log'
& $serverPath --version
if ($LASTEXITCODE -ne 0) { throw 'llama-server version check failed.' }
$helpText = (& $serverPath --help 2>&1 | Out-String)
foreach ($requiredFlag in @('--fit', '--no-context-shift', '--no-warmup', '--gpu-layers', '--flash-attn', '--alias')) {
    if (-not $helpText.Contains($requiredFlag)) { throw "llama-server build lacks $requiredFlag. Use a current official CUDA build." }
}
# Explicit fit=off prevents newer llama.cpp builds from changing GPU layers/context.
# No warmup means launching this server does not perform a hidden inference test.
# One slot preserves the requested context per query. f16 KV is recorded below.
$serverArgs = @('-m', ('"' + $modelPath + '"'), '--host', '127.0.0.1', '--port', $endpoint.Port,
    '--alias', $cfg.model, '--ctx-size', $cfg.context_window, '--parallel', '1',
    '--gpu-layers', '999', '--split-mode', 'none', '--main-gpu', '0', '--fit', 'off',
    '--flash-attn', 'on', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
    '--no-context-shift', '--no-warmup', '--batch-size', '512', '--ubatch-size', '128')
$serverProcess = Start-Process -FilePath $serverPath -ArgumentList $serverArgs -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
$serverStart = $serverProcess.StartTime.ToUniversalTime().ToString('o')
$ready = $false
for ($attempt = 0; $attempt -lt 180; $attempt++) {
    if ($serverProcess.HasExited) {
        $logText = Get-Content -LiteralPath $stderrPath -Raw
        $kind = if ($logText -match '(?i)out of memory|cuda.*alloc|failed to allocate') { 'HARDWARE_OOM' } else { 'SERVER_START_FAILED' }
        throw "$kind. Startup evidence preserved in $launchDir. No CPU fallback attempted."
    }
    try {
        $health = Invoke-RestMethod -Uri ($cfg.base_url + '/health') -TimeoutSec 2
        if ($health.status -eq 'ok') { $ready = $true; break }
    } catch { }
    Start-Sleep -Milliseconds 500
}
if (-not $ready) {
    Stop-Process -Id $serverProcess.Id
    throw 'Server startup timed out. Only the process started by this script was stopped.'
}
$startupLog = [IO.File]::ReadAllBytes($stderrPath)
$logString = [Text.Encoding]::UTF8.GetString($startupLog)
$layerMatches = [regex]::Matches($logString, 'offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU')
$allLayers = $layerMatches.Count -gt 0
if ($allLayers) {
    $last = $layerMatches[$layerMatches.Count - 1]
    $allLayers = [int]$last.Groups[1].Value -eq [int]$last.Groups[2].Value -and [int]$last.Groups[1].Value -gt 0
}
$kvGpu = $logString -match '(?i)CUDA\d+.*KV buffer|KV.*CUDA\d+.*buffer'
$cpuKv = $logString -match '(?i)CPU.*KV buffer|KV.*CPU.*buffer'
if (-not $allLayers -or -not $kvGpu -or $cpuKv) {
    Stop-Process -Id $serverProcess.Id
    throw 'GPU proof failed: require all model layers plus KV cache on CUDA. Startup logs preserved; process stopped.'
}
$props = Invoke-RestMethod -Uri ($cfg.base_url + '/props') -TimeoutSec 5
if ([int]$props.default_generation_settings.n_ctx -ne [int]$cfg.context_window) {
    Stop-Process -Id $serverProcess.Id
    throw 'Effective server context differs from configuration. Process stopped.'
}
$sha = [Security.Cryptography.SHA256]::Create()
$logHash = [Convert]::ToHexString($sha.ComputeHash($startupLog)).ToLowerInvariant()
$gpuInfo = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | Out-String).Trim()
$proof = [ordered]@{
    model = $cfg.model; context_window = [int]$cfg.context_window
    pid = $serverProcess.Id; process_start = $serverStart
    all_layers_gpu = $true; kv_gpu = $true; kv_type = 'f16'; gpu = $gpuInfo
    model_path = $modelPath; model_sha256 = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    executable_path = $serverPath; executable_sha256 = (Get-FileHash -LiteralPath $serverPath -Algorithm SHA256).Hash.ToLowerInvariant()
    log_path = $stderrPath; log_bytes = $startupLog.Length; log_sha256 = $logHash
    options = $serverArgs; quantization = $cfg.quantization
}
$proof | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $proofPath -Encoding utf8
$proof | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $launchDir 'gpu-proof.json') -Encoding utf8
Write-Host "Server ready at $($cfg.base_url), PID $($serverProcess.Id). All layers and KV verified on CUDA."
Write-Host "Proof: $proofPath"
Write-Host "Stop only this server with: Stop-Process -Id $($serverProcess.Id)"
Write-Host 'No benchmark inference performed. Startup/model loading is excluded from trial latency.'
