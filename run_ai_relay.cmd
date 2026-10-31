@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "CPA_RELAY_PYTHON=C:\miniconda\python.exe"
if exist "%CPA_RELAY_PYTHON%" goto ready
set "CPA_RELAY_PYTHON=python"
:ready
if "%~1"=="" goto interactive
"%CPA_RELAY_PYTHON%" "%~dp0.ai\relay.py" %*
exit /b %errorlevel%
:interactive
"%CPA_RELAY_PYTHON%" "%~dp0.ai\relay.py" --interactive
pause
