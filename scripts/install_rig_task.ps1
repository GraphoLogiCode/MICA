<#
Register (or remove) the MICA rig watcher as a Windows logon task -- the optional
fully-hands-off mode: log in, and the watcher waits in the background; launch
Minecraft and everything follows (rig, agent, post-session pipeline).

    powershell -File scripts\install_rig_task.ps1            # register at logon
    powershell -File scripts\install_rig_task.ps1 -DryRun    # show what would happen
    powershell -File scripts\install_rig_task.ps1 -Uninstall # remove the task

Notes (ASCII on purpose: Windows PowerShell 5.1 reads BOM-less files as ANSI):
- The task runs start-rig.bat in YOUR logon session, so the rig console window is
  visible (that is the point -- it is the rig's face). Close it or Ctrl+C to stop;
  it returns at next logon.
- Task Scheduler is deliberately the vehicle on this machine: processes started
  outside the normal desktop tree once hit broken NIO loopback selectors (see
  ISSUES 2026-07-04) -- a logon task runs in the ordinary user session where the
  whole toolchain is verified to work.
- The watcher itself holds a single-instance lock, so a manual start-rig.bat
  alongside the task is harmless: the second copy exits immediately.
#>
param(
    [switch]$Uninstall,
    [switch]$DryRun
)

$taskName = "MICA Rig Watcher"
$bat = Resolve-Path (Join-Path $PSScriptRoot "..\start-rig.bat")

if ($Uninstall) {
    if ($DryRun) { Write-Host "would remove scheduled task '$taskName'"; exit 0 }
    try {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop
        Write-Host "removed scheduled task '$taskName'"
    } catch {
        Write-Host "no task named '$taskName' was registered"
    }
    exit 0
}

$action = New-ScheduledTaskAction -Execute $bat.Path `
    -WorkingDirectory (Split-Path $bat.Path)
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
# No time limit (the watcher runs as long as the session does); one instance only --
# the watcher's own lock port is the second line of defense.
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew -StartWhenAvailable

if ($DryRun) {
    Write-Host "would register '$taskName': at logon of $env:USERNAME run $($bat.Path)"
    exit 0
}
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Force | Out-Null
Write-Host "registered '$taskName' -- the rig watcher starts at your next logon"
Write-Host "(remove with: powershell -File scripts\install_rig_task.ps1 -Uninstall)"
