$ErrorActionPreference = 'Stop'
$work = 'C:\Users\user\Documents\Codex\2026-10-06\task-2'
$telemetry = Join-Path $work 'host-live.json'
function Sample-Host {
  $memory = Get-CimInstance Win32_OperatingSystem
  $sample = @{epoch=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000;available_bytes=[long]$memory.FreePhysicalMemory*1024}
  $json = $sample | ConvertTo-Json -Compress
  [IO.File]::WriteAllText($telemetry+'.tmp',$json,[Text.UTF8Encoding]::new($false))
  Move-Item -LiteralPath ($telemetry+'.tmp') -Destination $telemetry -Force
  [IO.File]::AppendAllText((Join-Path $work 'host-audit.jsonl'),$json+"`n",[Text.UTF8Encoding]::new($false))
  return $sample
}
$initial=Sample-Host
if ($initial.available_bytes -lt 2GB) { throw 'Audit paused: host RAM below 2 GiB' }
$proc=Start-Process -FilePath 'wsl.exe' -ArgumentList @('--exec','nice','-n','15','/home/user/code/mia5150-local/.venv/bin/python','-I','-B','/mnt/c/Users/user/Documents/Codex/2026-10-06/task-2/run_audit.py') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $work 'audit.stdout.log') -RedirectStandardError (Join-Path $work 'audit.stderr.log')
while (!$proc.HasExited) {
  $null=Sample-Host
  Start-Sleep -Milliseconds 500
  $proc.Refresh()
}
$proc.WaitForExit()
Get-Content -LiteralPath (Join-Path $work 'audit.stdout.log')
Get-Content -LiteralPath (Join-Path $work 'audit.stderr.log')
if($proc.ExitCode -ne 0) {throw "Audit exited $($proc.ExitCode)"}
