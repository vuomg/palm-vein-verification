@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

REM ============================================================
REM SCA-MobileNet: 2 runs (3M + 12M), target EER < 0.9%%
REM ============================================================

set "DATASET=datasets/final_dataset_openset"
set "OUT=results/results_sca_final"

echo ============================================================
echo  SCA-MobileNet Final Run: 3M + 12M
echo  Started: %date% %time%
echo ============================================================

echo.
echo [1/2] SCA-3M (bottleneck, reduction=8) - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model sca_mobilenet --sca-backbone mobilenetv3 ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT%/sca_3m" ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 32 --epochs 100 --lr 0.001 ^
    --database default --eval-frequency 5 --seed 7
if errorlevel 1 (echo [FAILED] SCA-3M) else (echo [OK] SCA-3M)

echo.
echo [2/2] SCA-12M (no bottleneck, reduction=32) - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model sca_mobilenet --sca-backbone mobilenetv3 ^
    --no-bottleneck --ca-reduction 32 ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT%/sca_12m" ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 32 --epochs 100 --lr 0.001 ^
    --database default --eval-frequency 5 --seed 7

if errorlevel 1 (echo [FAILED] SCA-12M) else (echo [OK] SCA-12M)

echo.
echo ============================================================
echo  COMPLETE: %date% %time%
echo ============================================================
pause
