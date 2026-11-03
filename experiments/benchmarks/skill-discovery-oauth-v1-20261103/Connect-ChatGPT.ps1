[CmdletBinding()]
param([int]$Port = 1455, [switch]$NoOpenBrowser)
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run .\Setup.ps1 -InstallOAuth first.' }
Write-Host 'This must be completed in the browser on this Windows PC, not on iPhone.'
Write-Host 'Review permissions and storage, then click Continue with ChatGPT to approve the persistent connection.'
$readyPath = Join-Path $PSScriptRoot 'evidence\oauth-review.json'
if (Test-Path -LiteralPath $readyPath) {
    $ready = Get-Content -LiteralPath $readyPath -Raw | ConvertFrom-Json
    $reviewUri = [Uri]$ready.review_url
    if ($reviewUri.Scheme -eq 'http' -and $reviewUri.Host -eq '127.0.0.1' -and
        $reviewUri.AbsolutePath.StartsWith('/review/') -and
        [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() -lt [long]$ready.expires_at) {
        $active = $false
        try {
            $page = Invoke-WebRequest -Uri $reviewUri.AbsoluteUri -TimeoutSec 3
            $active = $page.StatusCode -eq 200 -and $page.Content.Contains('Continue with ChatGPT')
        } catch { }
        if ($active) {
            Write-Host 'Existing review listener reused. No second OAuth flow started.'
            Write-Host ('Valid until UTC: ' + [DateTimeOffset]::FromUnixTimeSeconds([long]$ready.expires_at).UtcDateTime.ToString('u'))
            if ($NoOpenBrowser) {
                Write-Host 'Browser opening skipped: preserve the pending Computer Use application approval.'
            } else {
                & $taskPython -c 'import sys,webbrowser; webbrowser.open(sys.argv[1])' $ready.review_url
                if ($LASTEXITCODE -ne 0) { throw 'Browser opening did not complete.' }
            }
            return
        }
    }
}
$loginArgs = @((Join-Path $PSScriptRoot 'oauth_chatgpt.py'), 'login', '--port', $Port, '--ready-file', $readyPath)
if (-not $NoOpenBrowser) { $loginArgs += '--open-browser' }
& $taskPython @loginArgs
if ($LASTEXITCODE -ne 0) { throw 'ChatGPT connection did not complete.' }
