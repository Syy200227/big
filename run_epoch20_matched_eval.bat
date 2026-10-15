@echo off
chcp 65001 >nul
setlocal
cd /d E:\workspace\EG-GM-PUNet
E:\anaconda\envs\torch221\python.exe evaluate_component_ablation_epoch20.py
set ERR=%ERRORLEVEL%
echo.
echo Result:
echo E:\workspace\EG-GM-PUNet\result\component_ablation_4_3_1_strict\epoch20_matched_evaluation\epoch20_ablation_table.csv
exit /b %ERR%
