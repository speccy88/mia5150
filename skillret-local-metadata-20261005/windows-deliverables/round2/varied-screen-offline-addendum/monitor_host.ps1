param([string]$OutputFile,[string]$StopFile)
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class HostRead {
 [StructLayout(LayoutKind.Sequential)] public struct FT {public uint lo; public uint hi; public ulong Value {get{return ((ulong)hi<<32)|lo;}}}
 [DllImport("kernel32.dll")] public static extern bool GetSystemTimes(out FT idle,out FT kernel,out FT user);
 [StructLayout(LayoutKind.Sequential)] public class MEM {public uint length=64;public uint load;public ulong totalPhys,availPhys,totalPage,availPage,totalVirtual,availVirtual,availExtended;}
 [DllImport("kernel32.dll",CharSet=CharSet.Auto)] public static extern bool GlobalMemoryStatusEx([In,Out] MEM m);
}
'@
$previousIdle = New-Object HostRead+FT
$previousKernel = New-Object HostRead+FT
$previousUser = New-Object HostRead+FT
[HostRead]::GetSystemTimes([ref]$previousIdle,[ref]$previousKernel,[ref]$previousUser) | Out-Null
$encoding = New-Object System.Text.UTF8Encoding($false)
$writer = New-Object System.IO.StreamWriter($OutputFile,$false,$encoding)
try {
 for ($iteration=0; $iteration -lt 43200 -and -not (Test-Path -LiteralPath $StopFile); $iteration++) {
  Start-Sleep -Seconds 1
  $nowIdle = New-Object HostRead+FT; $nowKernel = New-Object HostRead+FT; $nowUser = New-Object HostRead+FT
  [HostRead]::GetSystemTimes([ref]$nowIdle,[ref]$nowKernel,[ref]$nowUser) | Out-Null
  $deltaTotal=($nowKernel.Value-$previousKernel.Value)+($nowUser.Value-$previousUser.Value)
  $percent=100*(1-($nowIdle.Value-$previousIdle.Value)/[double]$deltaTotal)
  $memory = New-Object HostRead+MEM; [HostRead]::GlobalMemoryStatusEx($memory) | Out-Null
  $row=@{utc=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0;cpu_percent=$percent;available_bytes=$memory.availPhys;total_bytes=$memory.totalPhys}
  $writer.WriteLine(($row | ConvertTo-Json -Compress)); $writer.Flush()
  $previousIdle=$nowIdle; $previousKernel=$nowKernel; $previousUser=$nowUser
 }
} finally {$writer.Dispose()}

