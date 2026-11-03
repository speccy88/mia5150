[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ServerExe,
    [Parameter(Mandatory)][string]$ModelFile,
    [string]$Config = (Join-Path $PSScriptRoot 'configs\local-qwen3-06b.json')
)
$ErrorActionPreference = 'Stop'
function Read-SharedBytes([string]$Path) {
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    try {
        $memory = [IO.MemoryStream]::new()
        $stream.CopyTo($memory)
        return ,$memory.ToArray()
    } finally { $stream.Dispose(); if ($memory) { $memory.Dispose() } }
}
$serverPath = (Resolve-Path -LiteralPath $ServerExe).Path
$modelPath = (Resolve-Path -LiteralPath $ModelFile).Path
$configPath = (Resolve-Path -LiteralPath $Config).Path
$cfg = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if ($cfg.context_window -gt 32768) { throw 'PROFILE_CONTEXT_ABOVE_32768: no implicit native-window extension in this bounded lot.' }
if ($cfg.kv_type -notin @('f16','q8_0')) { throw 'This profile only permits verified f16 or q8_0 KV.' }
if ($cfg.backend -ne 'llama') { throw 'Config must use backend llama.' }
if ($cfg.model -notmatch '^[A-Za-z0-9._-]+$') { throw 'Use a simple model alias.' }
$endpoint = [Uri]$cfg.base_url
if ($endpoint.Host -ne '127.0.0.1' -or $endpoint.Scheme -ne 'http') { throw 'Server must bind to 127.0.0.1.' }
$runtimeDir = Join-Path $PSScriptRoot ('runtime\' + $cfg.model)
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$proofPath = Join-Path $runtimeDir 'gpu-proof.json'
$previousProof = $null
if (Test-Path -LiteralPath $proofPath) {
    $previousProof = Get-Content -LiteralPath $proofPath -Raw | ConvertFrom-Json -DateKind String
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
    '--flash-attn', 'on', '--cache-type-k', $cfg.kv_type, '--cache-type-v', $cfg.kv_type,
    '--no-context-shift', '--no-warmup', '--batch-size', '512', '--ubatch-size', '128',
    '--cache-ram', '0', '--ctx-checkpoints', '0', '--log-verbosity', '4')
$serverProcess = Start-Process -FilePath $serverPath -ArgumentList $serverArgs -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
$serverStart = $serverProcess.StartTime.ToUniversalTime().ToString('o')
[ordered]@{ pid=$serverProcess.Id; process_start=$serverStart; executable_path=$serverPath; model_path=$modelPath } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $launchDir 'owned-process.json') -Encoding utf8
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
$startupLog = Read-SharedBytes $stderrPath
$logString = [Text.Encoding]::UTF8.GetString($startupLog)
# WDDM may accept CUDA allocations beyond physical VRAM. Layer placement alone
# does not prove a physically feasible configured footprint.
$cudaBufferMatches = [regex]::Matches($logString, '(?im)CUDA0\s+(model|KV|RS|compute) buffer size\s*=\s*([0-9.]+) MiB')
$cudaBufferRecords = @($cudaBufferMatches | ForEach-Object {
    [ordered]@{kind=$_.Groups[1].Value; mib=[double]::Parse($_.Groups[2].Value,[Globalization.CultureInfo]::InvariantCulture); line=$_.Value}
})
$cudaStaticMiB = 0.0
$computeMiB = 0.0
foreach ($bufferRecord in $cudaBufferRecords) {
    if ($bufferRecord['kind'] -in @('model','KV','RS')) { $cudaStaticMiB += [double]$bufferRecord['mib'] }
    elseif ($bufferRecord['kind'] -eq 'compute') { $computeMiB = [Math]::Max($computeMiB,[double]$bufferRecord['mib']) }
}
$cudaStaticMiB += $computeMiB
$physicalVramText = (& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | Select-Object -First 1).Trim()
$physicalVramMiB = [double]::Parse($physicalVramText,[Globalization.CultureInfo]::InvariantCulture)
if ($cudaBufferMatches.Count -eq 0 -or $cudaStaticMiB -le 0) {
    Stop-Process -Id $serverProcess.Id -ErrorAction SilentlyContinue
    throw 'GPU_MEMORY_FOOTPRINT_UNVERIFIED: no positive CUDA buffer sum; no inference allowed.'
}
if ($cudaStaticMiB -gt $physicalVramMiB) {
    [ordered]@{kind='HARDWARE_VRAM_CAPACITY_EXCEEDED'; declared_cuda_buffers_mib=$cudaStaticMiB;
        physical_vram_mib=$physicalVramMiB; buffers=$cudaBufferRecords; inference_attempted=$false;
        interpretation='Declared CUDA footprint exceeds physical GPU memory; WDDM allocation success is not proof of residency.'} |
        ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $launchDir 'admission-failure.json') -Encoding utf8
    Stop-Process -Id $serverProcess.Id -ErrorAction SilentlyContinue
    throw "HARDWARE_VRAM_CAPACITY_EXCEEDED: $cudaStaticMiB MiB declared, $physicalVramMiB MiB physical. Evidence in $launchDir. No inference or CPU fallback."
}
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
$actualKvTypes = [regex]::Matches($logString, '(?:K|V) \(([^)]+)\):')
if ($actualKvTypes.Count -lt 2 -or @($actualKvTypes | Where-Object { $_.Groups[1].Value -ne $cfg.kv_type }).Count -gt 0) {
    Stop-Process -Id $serverProcess.Id
    throw 'KV_TYPE_NOT_VERIFIED: no silent cache fallback allowed.'
}
if ($logString -notmatch 'flash_attn\s*=\s*enabled') {
    Stop-Process -Id $serverProcess.Id
    throw 'FLASH_ATTENTION_NOT_VERIFIED'
}
$wddmSamples = @((Get-Counter '\GPU Process Memory(*)\*' -ErrorAction Stop).CounterSamples | Where-Object InstanceName -Like ('pid_' + $serverProcess.Id + '_*'))
$committedMiB = (($wddmSamples | Where-Object { $_.Path -Like '*\total committed' } | Measure-Object -Property CookedValue -Sum).Sum) / 1MB
if ($committedMiB -le 0 -or $committedMiB -gt $physicalVramMiB) {
    [ordered]@{kind='HARDWARE_WDDM_COMMIT_CAPACITY_EXCEEDED'; committed_mib=$committedMiB; physical_vram_mib=$physicalVramMiB; inference_attempted=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $launchDir 'admission-failure.json') -Encoding utf8
    Stop-Process -Id $serverProcess.Id
    throw 'WDDM process committed-memory admission failed before inference.'
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
    all_layers_gpu = $true; kv_gpu = $true; kv_type = $cfg.kv_type; gpu = $gpuInfo
    model_path = $modelPath; model_sha256 = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    executable_path = $serverPath; executable_sha256 = (Get-FileHash -LiteralPath $serverPath -Algorithm SHA256).Hash.ToLowerInvariant()
    log_path = $stderrPath; log_bytes = $startupLog.Length; log_sha256 = $logHash
    options = $serverArgs; quantization = $cfg.quantization
    wddm_committed_mib = $committedMiB; declared_cuda_buffers_mib = $cudaStaticMiB; physical_vram_mib = $physicalVramMiB
    cuda_buffer_records = $cudaBufferRecords
    residency_claim = 'Declared static buffers fit physical capacity; this is a necessary admission check, not a claim of zero CPU participation.'
}
$proof | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $proofPath -Encoding utf8
$proof | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $launchDir 'gpu-proof.json') -Encoding utf8
Write-Host "Server ready at $($cfg.base_url), PID $($serverProcess.Id). All layers and KV verified on CUDA."
Write-Host "Proof: $proofPath"
Write-Host "Stop only this server with: Stop-Process -Id $($serverProcess.Id)"
Write-Host 'No benchmark inference performed. Startup/model loading is excluded from trial latency.'
