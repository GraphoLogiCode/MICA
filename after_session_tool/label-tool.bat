@echo off
rem The one right way to open the labeling tool. Double-clicking gui.py hands it
rem to the py launcher's DEFAULT python (3.14 on this machine), which lacks the
rem pipeline's packages (torch/timm/gym3) — every evidence and label job fails.
rem This launcher pins the rig's python instead. Double-click me, not gui.py.
py -3.12 "%~dp0gui.py" %*
if errorlevel 1 pause
