@echo off
setlocal
cd /d "%~dp0.."
"C:\miniconda\envs\DLG\python.exe" -u -m qacpa.run_topk50_same80 --phase run
exit /b %errorlevel%
