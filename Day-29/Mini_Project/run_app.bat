@echo off
REM Double-click to open the OCR Document Reader with Python 3.11
cd /d "%~dp0"
py -3.11 main.py %*
if errorlevel 1 pause
