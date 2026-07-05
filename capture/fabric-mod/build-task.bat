@echo off
REM schtasks wrapper: the assistant tool environment cannot run Gradle in-tree
REM (NIO Selector.open fails inside its process tree), so builds run as a
REM scheduled task calling this file, which logs everything for the caller.
call D:\2026projects\MICA\capture\fabric-mod\build-mod.bat build > D:\2026projects\MICA\capture\fabric-mod\build-task.log 2>&1
