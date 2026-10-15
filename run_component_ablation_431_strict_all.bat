@echo off
chcp 65001 >nul
setlocal

cd /d E:\workspace\EG-GM-PUNet
if errorlevel 1 exit /b 1

echo ================================================================================
echo EG-GM-PUNet 4.3.1 STRICT MAIN COMPONENT ABLATION
echo Common V2-B init -^> MS-GMM / +MS-DEH / +EG-ALC
echo No A1a/A1final/A2/full checkpoint is used as formal initialization.
echo DMC stays ON and identical in all 3 rows.
echo Pipeline: preflight -^> common init -^> smoke -^> quick -^> formal -^> test -^> table
echo ================================================================================

E:\anaconda\envs\torch221\python.exe ^
  run_component_ablation_431_strict.py ^
  --config configs\component_ablation_431_strict.yaml

set ERR=%ERRORLEVEL%
echo.
echo ================================================================================
echo Exit code: %ERR%
echo Morning table:
echo E:\workspace\EG-GM-PUNet\result\component_ablation_4_3_1_strict\ablation_table_431_strict.csv
echo ================================================================================
exit /b %ERR%
