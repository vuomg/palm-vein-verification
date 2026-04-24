@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

REM ============================================================
REM Debug SCA-MobileNet: 2 variants on Internal Dataset
REM   1. 3M params  (with bottleneck)  - same config as old 0.89% run
REM   2. 12M params (without bottleneck) - old architecture
REM ============================================================

set "DATASET=datasets/final_dataset_openset"
set "OUT_BASE=results/results_sca_debug"
set FAIL_COUNT=0
set DONE_COUNT=0

echo ============================================================
echo  SCA-MobileNet Debug: 2 variants
echo  Config: batch-size 32, dropout 0.3, lr 0.001, 100 epochs
echo  Started: %date% %time%
echo ============================================================

echo.
echo [1/2] SCA-MobileNet 3M (with bottleneck) - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model sca_mobilenet --sca-backbone mobilenetv3 ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/sca_3m" ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 32 --epochs 100 --lr 0.001 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (echo [FAILED] SCA 3M & set /a FAIL_COUNT+=1) else (echo [OK] SCA 3M & set /a DONE_COUNT+=1)

echo.
echo [2/2] SCA-MobileNet 12M (no bottleneck) - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model sca_mobilenet --sca-backbone mobilenetv3 ^
    --no-bottleneck ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/sca_12m" ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 32 --epochs 100 --lr 0.001 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (echo [FAILED] SCA 12M & set /a FAIL_COUNT+=1) else (echo [OK] SCA 12M & set /a DONE_COUNT+=1)

echo.
echo ============================================================
echo  COMPLETE: %date% %time%
echo  Success: !DONE_COUNT!/2, Failed: !FAIL_COUNT!/2
echo ============================================================
pause
