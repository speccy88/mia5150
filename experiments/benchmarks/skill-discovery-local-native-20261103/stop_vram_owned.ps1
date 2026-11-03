param([ValidateSet('controller','bench','monitor')][string]$Kind)
$all=Get-CimInstance Win32_Process
if ($Kind -eq 'controller') { $ids=@(18512); $pattern='run_local_suite_v2.py' }
elseif ($Kind -eq 'bench') { $ids=@(14616,9336); $pattern='bench_native_v2.py' }
else { $ids=@(); $pattern='watch_local_progress.py' }
$selected=@($all | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match [regex]::Escape($pattern) -and ($_.CommandLine.Contains('task-5\local-benchmark') -or $_.ParentProcessId -in $ids) })
if ($Kind -eq 'bench') { $selected=@($selected | Where-Object { $_.CommandLine.Contains('qwen3-8b-q4km') }) }
foreach ($p in $selected) { Stop-Process -Id $p.ProcessId -ErrorAction SilentlyContinue }
$selected | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress
