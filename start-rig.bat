@echo off
rem MICA rig watcher: leave this window open, play singleplayer, press
rem Esc -> Open to LAN -- run_live.py + the MICA_AI agent (and FlowViz)
rem start themselves. Ctrl+C stops everything.
cd /d "%~dp0capture\mineflayer-bot"
node lan_autostart.js
pause
