@echo off
setlocal
cd /d "%~dp0.."
"C:\miniconda\envs\DLG\python.exe" -u -m qacpa.run_random_fill_same80 --config configs\cpa_fia_random_topk10_same80_R20_S5k_K3_F25k.json --phase run
exit /b %errorlevel%
