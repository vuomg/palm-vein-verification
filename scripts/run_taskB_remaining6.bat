@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

REM ============================================================
REM Task B (cont): 6 remaining models on Internal Dataset
REM Priority 1: Models missing from paper (---)
REM Priority 2: Models with EER in paper but no metrics file
REM ============================================================

set "DATASET=datasets/final_dataset_openset"
set "OUT_BASE=results/results_taskB_internal"
set FAIL_COUNT=0
set DONE_COUNT=0

echo ============================================================
echo  TASK B continued: 6 remaining models
echo  Started: %date% %time%
echo ============================================================

REM ============================================================
REM PRIORITY 1: Models NOT in paper (--- in SOTA table)
REM ============================================================

echo.
echo [1/6] MPSNet - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model mpsnet ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/mpsnet" ^
    --feature-dim 1024 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (echo [FAILED] MPSNet & set /a FAIL_COUNT+=1) else (echo [OK] MPSNet & set /a DONE_COUNT+=1)

echo.
echo [2/6] EfficientNet-B0 - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model sca_mobilenet --sca-backbone efficientnet_b0 ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/efficientnet_b0" ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (echo [FAILED] EfficientNet-B0 & set /a FAIL_COUNT+=1) else (echo [OK] EfficientNet-B0 & set /a DONE_COUNT+=1)

echo.
echo [3/6] ResNet-50 - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model gscl --gscl-backbone resnet50 ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/resnet50" ^
    --feature-dim 1024 ^
    --batch-size 64 --epochs 100 --lr 0.01 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (echo [FAILED] ResNet-50 & set /a FAIL_COUNT+=1) else (echo [OK] ResNet-50 & set /a DONE_COUNT+=1)

REM ============================================================
REM PRIORITY 2: Models with EER in paper but no metrics file
REM ============================================================

echo.
echo [4/6] RSNet - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model rsnet ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/rsnet" ^
    --feature-dim 1024 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (echo [FAILED] RSNet & set /a FAIL_COUNT+=1) else (echo [OK] RSNet & set /a DONE_COUNT+=1)

echo.
echo [5/6] FGFNet - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model fgfnet ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/fgfnet" ^
    --feature-dim 640 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (echo [FAILED] FGFNet & set /a FAIL_COUNT+=1) else (echo [OK] FGFNet & set /a DONE_COUNT+=1)

echo.
echo [6/6] GSCL ResNet-18 - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model gscl --gscl-backbone resnet18 ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/gscl_resnet18" ^
    --feature-dim 512 ^
    --batch-size 64 --epochs 100 --lr 0.01 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (echo [FAILED] GSCL ResNet-18 & set /a FAIL_COUNT+=1) else (echo [OK] GSCL ResNet-18 & set /a DONE_COUNT+=1)

REM ============================================================
echo.
echo ============================================================
echo  ALL 6 MODELS COMPLETE
echo  Finished: %date% %time%
echo  Success: !DONE_COUNT!/6, Failed: !FAIL_COUNT!/6
echo ============================================================
