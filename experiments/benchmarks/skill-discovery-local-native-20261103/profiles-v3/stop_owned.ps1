param([int]$PidToStop,[string]$Exe,[string]$Started)
$p=Get-Process -Id $PidToStop -ErrorAction SilentlyContinue
if ($p) { if ($p.Path -ne $Exe -or $p.StartTime.ToUniversalTime().ToString('o') -ne $Started) { exit 3 }; Stop-Process -Id $p.Id }
