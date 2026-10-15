@echo off
chcp 65001 >nul
setlocal

cd /d E:\workspace\EG-GM-PUNet
if errorlevel 1 exit /b 1

echo ================================================================================
echo LAST OPTIONAL PAPER RUN
echo Full EG-GM-PUNet w/o DMC - strict common init - 20 epochs
echo Smoke -> Formal -> Test evaluation
echo ================================================================================
echo.

E:\anaconda\envs\torch221\python.exe run_full_egalc_no_dmc_epoch20.py

set ERR=%ERRORLEVEL%
echo.
echo ================================================================================
echo Exit code: %ERR%
echo Output root:
echo E:\workspace\EG-GM-PUNet\result\component_ablation_4_3_1_strict\full_egalc_no_dmc
echo ================================================================================
exit /b %ERR%
