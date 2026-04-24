@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

REM ============================================================
REM SCA-MobileNet: 5 seeds x 2 variants = 10 runs
REM   3M:  with bottleneck, ca_reduction=8
REM   12M: no bottleneck, ca_reduction=32
REM Seeds: 42, 0, 1, 7, 99
REM ============================================================

set "DATASET=datasets/final_dataset_openset"
set FAIL_COUNT=0
set DONE_COUNT=0
set TOTAL=10

echo ============================================================
echo  SCA-MobileNet 5-seed experiment
echo  Config: batch-size 32, dropout 0.3, lr 0.001, 100 epochs
echo  Started: %date% %time%
echo ============================================================

REM --- 3M variant (with bottleneck, reduction=8) ---

for %%S in (42 0 1 7 99) do (
    echo.
    echo [!DONE_COUNT!/%TOTAL%] 3M seed=%%S - Started: %time%
    echo --------------------------------------------------------
    python train.py ^
        --model sca_mobilenet --sca-backbone mobilenetv3 ^
        --dataset "%DATASET%" ^
        --output-dir "results/results_sca_3m_s%%S" ^
        --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
        --batch-size 32 --epochs 100 --lr 0.001 ^
        --database default --eval-frequency 5 --seed %%S
    if errorlevel 1 (echo [FAILED] 3M s%%S & set /a FAIL_COUNT+=1) else (echo [OK] 3M s%%S & set /a DONE_COUNT+=1)
)

REM --- 12M variant (no bottleneck, reduction=32) ---

for %%S in (42 0 1 7 99) do (
    echo.
    echo [!DONE_COUNT!/%TOTAL%] 12M seed=%%S - Started: %time%
    echo --------------------------------------------------------
    python train.py ^
        --model sca_mobilenet --sca-backbone mobilenetv3 ^
        --no-bottleneck --ca-reduction 32 ^
        --dataset "%DATASET%" ^
        --output-dir "results/results_sca_12m_s%%S" ^
        --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
        --batch-size 32 --epochs 100 --lr 0.001 ^
        --database default --eval-frequency 5 --seed %%S
    if errorlevel 1 (echo [FAILED] 12M s%%S & set /a FAIL_COUNT+=1) else (echo [OK] 12M s%%S & set /a DONE_COUNT+=1)
)

echo.
echo ============================================================
echo  COMPLETE: %date% %time%
echo  Success: !DONE_COUNT!/%TOTAL%, Failed: !FAIL_COUNT!/%TOTAL%
echo ============================================================
pause
