<#
Launch a long-running chain so it survives Claude Code and the terminal that started it.

A process started with Start-Process (or as a background task) from an app's shell is
that shell's child, and it can be killed when the app's session ends: every chain
died at 00:15 on 16 October 2026 with no error and no reboot. A process created
through WMI (Win32_Process.Create) is a child of the WMI service instead, so it runs
until it finishes, the PC restarts or someone stops it.

    powershell -ExecutionPolicy Bypass -File scripts\launch_detached.ps1 runs\finetune\chain.cmd

The chain's output is appended to chain.log beside the .cmd file. The launcher refuses
to start a chain that is already running: two copies would train the same checkpoints.
#>
param([Parameter(Mandatory = $true)][string]$Chain)

$root = Split-Path -Parent $PSScriptRoot
$path = if ([IO.Path]::IsPathRooted($Chain)) { $Chain } else { Join-Path $root $Chain }
$path = (Resolve-Path $path).Path
if ($path -match ' ') { throw "path contains a space, which cmd /c would split: $path" }

$running = Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" |
    Where-Object { $_.CommandLine -like "*$path*" }
if ($running) { throw "already running as pid $(($running.ProcessId) -join ', '): $path" }

$log = [IO.Path]::ChangeExtension($path, '.log')
# Start hidden (SW_HIDE): otherwise Windows 11 hands each chain to Windows Terminal as
# a visible tab, and closing that tab kills the chain.
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
    -Arguments @{ CommandLine = "cmd.exe /c $path >> $log 2>&1"; CurrentDirectory = $root;
                  ProcessStartupInformation = $startup }
if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create returned $($r.ReturnValue) for $path" }
"started $path as pid $($r.ProcessId) (detached, under the WMI service); log $log"
