@echo off
REM Build helper: pins the portable JDK 17 and runs the Fabric build from this dir.
set "JAVA_HOME=D:\2026projects\MICA\capture\.toolchain\jdk-17.0.19+10"
cd /d "D:\2026projects\MICA\capture\fabric-mod"
call gradlew.bat %*
