@echo off
REM Backs up the irreplaceable research corpus to a second disk (added 2026-07-12).
REM   backup-corpus.bat [destination]     (default: F:\MICA-backup)
REM
REM Additive on purpose: /E copies subdirs, /XO skips files the backup already has
REM at the same or newer age. NEVER /MIR here — a mistyped destination with /MIR
REM would DELETE whatever it finds there. Robocopy exit codes 0-7 mean success
REM (0 = nothing new, 1 = files copied); 8+ is a real failure.
setlocal
set "DEST=%~1"
if "%DEST%"=="" set "DEST=F:\MICA-backup"
if not exist "%DEST%" mkdir "%DEST%"

echo Backing up the MICA corpus to %DEST%
robocopy "D:\2026projects\MICA\capture\raw" "%DEST%\capture-raw" /E /XO /R:2 /W:5 /NFL /NDL /NP /LOG+:"%DEST%\backup.log"
if errorlevel 8 goto :failed
robocopy "D:\2026projects\MICA\models" "%DEST%\models" /E /XO /R:2 /W:5 /NFL /NDL /NP /LOG+:"%DEST%\backup.log"
if errorlevel 8 goto :failed
robocopy "C:\Users\ThinkLogiCode\Documents\Obsidian Vault\MICA Research" "%DEST%\vault" /E /XO /R:2 /W:5 /NFL /NDL /NP /LOG+:"%DEST%\backup.log"
if errorlevel 8 goto :failed

echo Backup complete. Log: %DEST%\backup.log
exit /b 0

:failed
echo BACKUP FAILED (robocopy exit %errorlevel%) - see %DEST%\backup.log
exit /b 1
