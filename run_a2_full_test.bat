@echo off
setlocal
cd /d E:\workspace\EG-GM-PUNet

echo ============================================================
echo A2 TEST 1/2: epoch12 best GED / best distribution score
echo ============================================================
E:\anaconda\envs\torch221\python.exe ^
  evaluate\evaluate_eg_gm_punet_v3_a2.py ^
  --config configs\eg_gm_punet_v3_a2_direct_diversity_match_lidc.yaml ^
  --checkpoint result\eg_gm_punet_v3_a2_direct_diversity_match_k5\best_val_ged4_eg_gm_punet_v3_a2.pth ^
  --split test ^
  --num-samples 16 ^
  --sample-counts 1 4 8 16
if errorlevel 1 goto :error

echo ============================================================
echo A2 TEST 2/2: epoch24 minimum diversity-gap checkpoint
echo ============================================================
E:\anaconda\envs\torch221\python.exe ^
  evaluate\evaluate_eg_gm_punet_v3_a2.py ^
  --config configs\eg_gm_punet_v3_a2_direct_diversity_match_lidc.yaml ^
  --checkpoint result\eg_gm_punet_v3_a2_direct_diversity_match_k5\best_val_distribution_gap_eg_gm_punet_v3_a2.pth ^
  --split test ^
  --num-samples 16 ^
  --sample-counts 1 4 8 16
if errorlevel 1 goto :error

echo.
echo ============================================================
echo BOTH A2 FULL TESTS FINISHED
echo ============================================================
goto :eof

:error
echo.
echo [ERROR] Evaluation stopped. Please send the full traceback.
exit /b 1
