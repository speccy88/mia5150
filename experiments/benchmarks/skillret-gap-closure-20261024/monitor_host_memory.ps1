#requires -Version 7.0
param([string]$OutputDirectory=([IO.DirectoryInfo]$PSScriptRoot).Parent.Parent.Parent.Parent.FullName)
$ErrorActionPreference='Stop'
$outputFile=Join-Path $OutputDirectory 'host-memory.json'
$stopFile=Join-Path $OutputDirectory 'host-monitor.stop'
while (-not (Test-Path -LiteralPath $stopFile)) {
    $os=Get-CimInstance Win32_OperatingSystem
    $sample=[ordered]@{total_bytes=([int64]$os.TotalVisibleMemorySize*1024);utc=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000;available_bytes=([int64]$os.FreePhysicalMemory*1024)}
    $temporary=Join-Path $OutputDirectory 'host-memory.next.json'
    [IO.File]::WriteAllText($temporary,($sample | ConvertTo-Json -Compress))
    Move-Item -LiteralPath $temporary -Destination $outputFile -Force
    Start-Sleep -Seconds 2
}
