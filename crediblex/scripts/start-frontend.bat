@echo off
REM Windows compatibility script to bypass execution policies
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-frontend.ps1"
