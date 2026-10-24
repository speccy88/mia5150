#requires -Version 7.0
param(
    [Parameter(Mandatory=$true)][string]$StudyPath,
    [Parameter(Mandatory=$true)][string]$RuntimePath,
    [Parameter(Mandatory=$true)][string]$LinuxPython,
    [string]$Distribution='Ubuntu-26.04'
)
$ErrorActionPreference='Stop'
# No prompt or new key file is created. The user-authorized encrypted file must
# already exist under the same Windows identity that encrypted it.
$credentialPath=Join-Path $env:TEMP 'mia5150-openrouter-key.dpapi'
if (-not (Test-Path -LiteralPath $credentialPath -PathType Leaf)) {
    throw 'Expected same-user encrypted credential is absent; no hosted call started.'
}
$studyLinux=(& wsl.exe -d $Distribution -- wslpath -a $StudyPath).Trim()
$runtimeLinux=(& wsl.exe -d $Distribution -- wslpath -a $RuntimePath).Trim()
$privateLogPath=Join-Path $RuntimePath 'hosted-private'
[IO.Directory]::CreateDirectory($privateLogPath) | Out-Null
$launch=New-Object Diagnostics.ProcessStartInfo
$launch.FileName='wsl.exe'
$launch.UseShellExecute=$false
$launch.CreateNoWindow=$true
$launch.RedirectStandardInput=$true
$launch.RedirectStandardOutput=$true
$launch.RedirectStandardError=$true
foreach ($argument in @('-d',$Distribution,'--',$LinuxPython,($studyLinux+'/hosted/runner.py'),'--runtime-root',$runtimeLinux)) {
    $launch.ArgumentList.Add($argument)
}
$benchmarkProcess=New-Object Diagnostics.Process
$benchmarkProcess.StartInfo=$launch
$secretPointer=[IntPtr]::Zero
$transferred=$false
try {
    $protectedValue=Get-Content -LiteralPath $credentialPath -Raw | ConvertTo-SecureString
    $secretPointer=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($protectedValue)
    $memoryCredential=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($secretPointer)
    if (-not $benchmarkProcess.Start()) { throw 'Unable to start bounded runner.' }
    $stdoutTask=$benchmarkProcess.StandardOutput.ReadToEndAsync()
    $stderrTask=$benchmarkProcess.StandardError.ReadToEndAsync()
    $benchmarkProcess.StandardInput.Write($memoryCredential)
    $benchmarkProcess.StandardInput.Close()
    $transferred=$true
} finally {
    if ($secretPointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($secretPointer) }
    $memoryCredential=$null
    if ($protectedValue) { $protectedValue.Dispose() }
    # The user explicitly revoked automatic deletion on 24 October 2026.
    # Preserve the encrypted local copy until explicitly requested otherwise.
}
if ($transferred) {
    $benchmarkProcess.WaitForExit()
    [IO.File]::WriteAllText((Join-Path $privateLogPath 'runner-stdout.txt'),$stdoutTask.GetAwaiter().GetResult())
    [IO.File]::WriteAllText((Join-Path $privateLogPath 'runner-stderr.txt'),$stderrTask.GetAwaiter().GetResult())
    [ordered]@{runner_exit_code=$benchmarkProcess.ExitCode;encrypted_copy_removed=(-not (Test-Path -LiteralPath $credentialPath));credential_transport='anonymous process stdin';credential_printed=$false} | ConvertTo-Json
}
