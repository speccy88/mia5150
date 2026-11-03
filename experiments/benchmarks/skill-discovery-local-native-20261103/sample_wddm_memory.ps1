param([Parameter(Mandatory)][string]$ProofPath)
$ErrorActionPreference = 'Stop'
$record = Get-Content -LiteralPath $ProofPath -Raw | ConvertFrom-Json -DateKind String
$owned = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
if (-not $owned) { exit 2 }
if ($owned.Path -ne $record.executable_path -or $owned.StartTime.ToUniversalTime().ToString('o') -ne $record.process_start) { exit 3 }
$counters = (Get-Counter '\GPU Process Memory(*)\*' -ErrorAction Stop).CounterSamples |
    Where-Object InstanceName -Like ('pid_' + $record.pid + '_*')
$values = @($counters | ForEach-Object {
    [ordered]@{instance=$_.InstanceName; counter=($_.Path -split '\\')[-1]; bytes=$_.CookedValue; status=$_.Status}
})
[ordered]@{utc=[DateTime]::UtcNow.ToString('o'); model=$record.model; pid=$record.pid; samples=$values} |
    ConvertTo-Json -Depth 5 -Compress
