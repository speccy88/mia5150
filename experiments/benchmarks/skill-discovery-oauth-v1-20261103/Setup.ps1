[CmdletBinding()]
param([switch]$InstallOAuth, [switch]$InstallPlots)
$ErrorActionPreference = 'Stop'
$taskDir = $PSScriptRoot
$pythonCandidates = @(
    (Join-Path $env:LOCALAPPDATA 'Python\pythoncore-3.12-64\python.exe'),
    (Join-Path $env:USERPROFILE '.venvs\mia5150-a2\Scripts\python.exe')
)
$taskPython = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $taskPython) { throw 'Python 3.12 required. Set up Python, then rerun this script.' }
$venvPython = Join-Path $taskDir '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $taskPython -m venv --without-pip (Join-Path $taskDir '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Local virtual environment creation failed.' }
}
if ($InstallOAuth -or $InstallPlots) {
    $setupTemp = Join-Path $taskDir '.setup-tmp'
    New-Item -ItemType Directory -Path $setupTemp -Force | Out-Null
    & $venvPython -c 'import tempfile,sys,ensurepip; tempfile.tempdir=sys.argv[1]; ensurepip.bootstrap(upgrade=True)' $setupTemp
    if ($LASTEXITCODE -ne 0) { throw 'Local pip setup failed.' }
    $requirementFiles = @()
    if ($InstallOAuth) { $requirementFiles += 'requirements-oauth.txt' }
    if ($InstallPlots) { $requirementFiles += 'requirements-plots.txt' }
    foreach ($requirementFile in $requirementFiles) {
        & $venvPython -c 'import tempfile,sys,runpy; tempfile.tempdir=sys.argv.pop(1); runpy.run_module("pip",run_name="__main__")' $setupTemp install --disable-pip-version-check -r (Join-Path $taskDir $requirementFile)
        if ($LASTEXITCODE -ne 0) { throw 'Optional dependency installation failed.' }
    }
}
& $venvPython (Join-Path $taskDir 'bench.py') validate
if ($LASTEXITCODE -ne 0) { throw 'Dataset validation failed.' }
Write-Host 'Environment ready. No model inference or OAuth login performed.'
