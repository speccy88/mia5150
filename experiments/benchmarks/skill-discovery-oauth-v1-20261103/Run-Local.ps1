[CmdletBinding()]
param(
    [string]$Config = (Join-Path $PSScriptRoot 'configs\local-qwen3-06b.json'),
    [string]$Out = (Join-Path $PSScriptRoot 'results\local-qwen3-06b'),
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run .\Setup.ps1 first.' }
$runArgs = @((Join-Path $PSScriptRoot 'bench.py'), 'run', '--config', $Config, '--out', $Out)
if ($Resume) { $runArgs += '--resume' }
& $taskPython @runArgs
if ($LASTEXITCODE -ne 0) { throw 'Benchmark stopped; preserve results and inspect the local error.' }
