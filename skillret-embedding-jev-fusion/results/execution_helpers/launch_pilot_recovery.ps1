$ErrorActionPreference = 'Stop'
$workspace = 'C:\Users\user\Documents\Codex\2026-10-06\task-3'
$experiment = '/home/user/code/mia5150-local/skillret-embedding-jev-fusion'
$pythonExecutable = '/home/user/code/mia5150-local/.venv/bin/python'
$wslWorkspace = '/mnt/c/Users/user/Documents/Codex/2026-10-06/task-3'
$outPath = Join-Path $workspace 'pilot-recovery-launcher.stdout.log'
$errPath = Join-Path $workspace 'pilot-recovery-launcher.stderr.log'
$receiptPath = Join-Path $workspace 'pilot-recovery-launcher.json'
if ((Test-Path -LiteralPath $receiptPath) -or (Test-Path -LiteralPath $outPath) -or (Test-Path -LiteralPath $errPath)) { throw 'Use a reviewed new launch; existing launcher evidence is preserved.' }
$arguments = @('-d','Ubuntu-26.04','--',$pythonExecutable,"$experiment/resource_guard.py",'--host',"$wslWorkspace/skillret-embedding-jev-fusion/hosted-telemetry-20261007-0158.jsonl",'--run-dir',"$experiment/runs/13-hosted-pilot-reviewed-recovery",'--',$pythonExecutable,"$wslWorkspace/execute_authorized_jev_study.py",'--plan',"$experiment/results/jev-plan-pilot-jev113-context-v1/preparation_manifest.json",'--gates',"$experiment/results/hosted-execution/recovery-after-wsl-restart/pilot_recovery_gate_config_v2.json")
$process = Start-Process -FilePath 'C:\Windows\System32\wsl.exe' -ArgumentList $arguments -WindowStyle Hidden -RedirectStandardOutput $outPath -RedirectStandardError $errPath -PassThru
$receipt = @{ launched_at_unix=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds(); process_id=$process.Id; hidden=$true; arguments=$arguments; original_first_attempt_deadline_unchanged=$true; key_value_not_in_arguments=$true }
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
$receipt | ConvertTo-Json -Compress -Depth 5
