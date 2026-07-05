@echo off
REM Build helper: pins the portable BUILD JDK and runs the Fabric build from this dir.
REM JDK 21 since 2026-07-04: fabric-loom 1.17.12+ requires a Java 21 build JVM (the
REM produced mod still targets MC 1.16.5's bytecode; the game's runtime is untouched).
REM The wrapper is called by explicit path: some shells (hardened PATH lookup) refuse
REM to resolve bare .bat names from the current directory.
set "JAVA_HOME=D:\2026projects\MICA\capture\.toolchain\jdk-21.0.11+10"
cd /d "D:\2026projects\MICA\capture\fabric-mod"
call "%~dp0gradlew.bat" %*
