@echo off
setlocal
cd /d "%~dp0.."
"C:\miniconda\envs\DLG\python.exe" -u -m qacpa.queue_topk_ratios_same80 --phase run
exit /b %errorlevel%
