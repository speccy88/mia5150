# USER-RUN ONLY, after direct approval is recorded in this existing Codex task.
# Preparing this script does not grant approval or resolve the review rejection.
[CmdletBinding()]
param([switch]$AfterDirectTaskApproval)
$ErrorActionPreference = 'Stop'
if (-not $AfterDirectTaskApproval) {
    throw 'Blocked: direct user approval in the existing task must resolve the API authorization review first. This flag cannot substitute for that approval.'
}
$workspace = Split-Path -Parent $PSScriptRoot
$monitor = Join-Path $workspace 'monitor_host.ps1'
if (-not (Test-Path -LiteralPath $monitor -PathType Leaf)) { throw 'Existing host monitor missing.' }
$runId = [Guid]::NewGuid().ToString('N')
$telemetry = Join-Path $PSScriptRoot ("varied-user-host-$runId.jsonl")
$stopFile = Join-Path $PSScriptRoot ("varied-user-host-$runId.stop")
Write-Host 'TypeSafe Jev, https://api.typesafe.ai/v1/systemone; saved WSL Jev key.'
Write-Host 'Outgoing PUBLIC SkillRet query text and candidate names/descriptions only; no labels or private files.'
Write-Host 'Maximum 30 POST calls / 1.92M input tokens / USD0.08064 metered usage; USD0.10 reserved within shared USD4 cap. No top-up, retries or other provider.'
$job = Start-Job -FilePath $monitor -ArgumentList $telemetry,$stopFile
$workerExit = 1
try {
    $deadline = [DateTime]::UtcNow.AddSeconds(15)
    while (-not (Test-Path -LiteralPath $telemetry)) {
        if ($job.State -in @('Failed','Completed','Stopped') -or [DateTime]::UtcNow -gt $deadline) { throw 'Host monitor unavailable; no worker launched.' }
        Start-Sleep -Milliseconds 500
    }
    Start-Sleep -Seconds 2
    $latest = Get-Content -LiteralPath $telemetry -Tail 1 | ConvertFrom-Json
    if (-not $latest -or [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - $latest.utc -gt 8) { throw 'Host telemetry stale; no worker launched.' }
    $linuxTelemetry = (& wsl.exe -d Ubuntu-26.04 -- wslpath -u $telemetry).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $linuxTelemetry.StartsWith('/mnt/c/')) { throw 'Could not resolve telemetry path.' }
    & wsl.exe -d Ubuntu-26.04 -- env "MIA_HOST_TELEMETRY=$linuxTelemetry" python3 /home/user/code/mia5150-local/guard.py varied-screen-v3 /home/user/code/mia5150-local/.venv/bin/python /home/user/code/mia5150-local/round2/varied_screen.py
    $workerExit = $LASTEXITCODE
} finally {
    Set-Content -LiteralPath $stopFile -Value 'stop'
    Wait-Job -Job $job -Timeout 5 | Out-Null
    if ($job.State -eq 'Running') { Stop-Job -Job $job }
    Receive-Job -Job $job | Out-Null
    Remove-Job -Job $job
}
exit $workerExit
