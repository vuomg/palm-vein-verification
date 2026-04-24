@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

REM ============================================================
REM Task B: Train 12 Models on Internal Dataset (1549 IDs)
REM Output: results/results_taskB_internal/<model>/
REM Estimated: ~8-10 hours on RTX 4050 Laptop
REM ============================================================

set "DATASET=datasets/final_dataset_openset"
set "OUT_BASE=results/results_taskB_internal"
set FAIL_COUNT=0
set DONE_COUNT=0

echo ============================================================
echo  TASK B: 12 Models on Internal Dataset (1549 IDs)
echo  Dataset: %DATASET%
echo  Output:  %OUT_BASE%
echo  Started: %date% %time%
echo ============================================================

REM Check dataset exists
if not exist "%DATASET%\train" (
    echo [ERROR] Dataset not found: %DATASET%\train
    pause
    exit /b 1
)

REM Create output base
if not exist "%OUT_BASE%" mkdir "%OUT_BASE%"

REM ============================================================
REM 1/12: SCA-MobileNet (Ours)
REM ============================================================
echo.
echo [1/12] SCA-MobileNet - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model sca_mobilenet --sca-backbone mobilenetv3 ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/sca_mobilenet" ^
    --loss-type adacos_only --feature-dim 1024 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (
    echo [FAILED] SCA-MobileNet
    set /a FAIL_COUNT+=1
) else (
    echo [OK] SCA-MobileNet
    set /a DONE_COUNT+=1
)

REM ============================================================
REM 2/12: MobileNetV3 Base (no STN, no CA, no SPP)
REM ============================================================
echo.
echo [2/12] MobileNetV3 Base - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model sca_mobilenet --sca-backbone mobilenetv3 ^
    --no-stn --no-ca --no-spp ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/mobilenetv3_base" ^
    --loss-type adacos_only --feature-dim 1024 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (
    echo [FAILED] MobileNetV3 Base
    set /a FAIL_COUNT+=1
) else (
    echo [OK] MobileNetV3 Base
    set /a DONE_COUNT+=1
)

REM ============================================================
REM 3/12: MPSNet
REM ============================================================
echo.
echo [3/12] MPSNet - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model mpsnet ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/mpsnet" ^
    --feature-dim 1024 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (
    echo [FAILED] MPSNet
    set /a FAIL_COUNT+=1
) else (
    echo [OK] MPSNet
    set /a DONE_COUNT+=1
)

REM ============================================================
REM 4/12: EfficientNet-B0
REM ============================================================
echo.
echo [4/12] EfficientNet-B0 - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model sca_mobilenet --sca-backbone efficientnet_b0 ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/efficientnet_b0" ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (
    echo [FAILED] EfficientNet-B0
    set /a FAIL_COUNT+=1
) else (
    echo [OK] EfficientNet-B0
    set /a DONE_COUNT+=1
)

REM ============================================================
REM 5/12: DeiT-Tiny
REM ============================================================
echo.
echo [5/12] DeiT-Tiny - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model sca_mobilenet --sca-backbone deit_tiny ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/deit_tiny" ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (
    echo [FAILED] DeiT-Tiny
    set /a FAIL_COUNT+=1
) else (
    echo [OK] DeiT-Tiny
    set /a DONE_COUNT+=1
)

REM ============================================================
REM 6/12: MobileViT-S
REM ============================================================
echo.
echo [6/12] MobileViT-S - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model sca_mobilenet --sca-backbone mobilevit_s ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/mobilevit_s" ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (
    echo [FAILED] MobileViT-S
    set /a FAIL_COUNT+=1
) else (
    echo [OK] MobileViT-S
    set /a DONE_COUNT+=1
)

REM ============================================================
REM 7/12: Swin-Tiny
REM ============================================================
echo.
echo [7/12] Swin-Tiny - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model sca_mobilenet --sca-backbone swin_tiny ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/swin_tiny" ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (
    echo [FAILED] Swin-Tiny
    set /a FAIL_COUNT+=1
) else (
    echo [OK] Swin-Tiny
    set /a DONE_COUNT+=1
)

REM ============================================================
REM 8/12: RSNet
REM ============================================================
echo.
echo [8/12] RSNet - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model rsnet ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/rsnet" ^
    --feature-dim 1024 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (
    echo [FAILED] RSNet
    set /a FAIL_COUNT+=1
) else (
    echo [OK] RSNet
    set /a DONE_COUNT+=1
)

REM ============================================================
REM 9/12: FGFNet
REM ============================================================
echo.
echo [9/12] FGFNet - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model fgfnet ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/fgfnet" ^
    --feature-dim 640 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (
    echo [FAILED] FGFNet
    set /a FAIL_COUNT+=1
) else (
    echo [OK] FGFNet
    set /a DONE_COUNT+=1
)

REM ============================================================
REM 10/12: GSCL (ResNet-18)
REM ============================================================
echo.
echo [10/12] GSCL ResNet-18 - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model gscl --gscl-backbone resnet18 ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/gscl_resnet18" ^
    --feature-dim 512 ^
    --batch-size 64 --epochs 100 --lr 0.01 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (
    echo [FAILED] GSCL ResNet-18
    set /a FAIL_COUNT+=1
) else (
    echo [OK] GSCL ResNet-18
    set /a DONE_COUNT+=1
)

REM ============================================================
REM 11/12: ResNet-50 (via GSCL framework)
REM ============================================================
echo.
echo [11/12] ResNet-50 - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model gscl --gscl-backbone resnet50 ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/resnet50" ^
    --feature-dim 1024 ^
    --batch-size 64 --epochs 100 --lr 0.01 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (
    echo [FAILED] ResNet-50
    set /a FAIL_COUNT+=1
) else (
    echo [OK] ResNet-50
    set /a DONE_COUNT+=1
)

REM ============================================================
REM 12/12: Modified-DenseNet161 (EUSIPCO 2020)
REM ============================================================
echo.
echo [12/12] Modified-DenseNet161 - Started: %time%
echo --------------------------------------------------------
python train.py ^
    --model eusipco2020 ^
    --dataset "%DATASET%" ^
    --output-dir "%OUT_BASE%/densenet161" ^
    --feature-dim 1024 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database default --eval-frequency 5 --seed 42
if errorlevel 1 (
    echo [FAILED] Modified-DenseNet161
    set /a FAIL_COUNT+=1
) else (
    echo [OK] Modified-DenseNet161
    set /a DONE_COUNT+=1
)

REM ============================================================
REM SUMMARY
REM ============================================================
echo.
echo ============================================================
echo  TASK B COMPLETE
echo  Finished: %date% %time%
echo  Success: !DONE_COUNT!/12, Failed: !FAIL_COUNT!/12
echo ============================================================
echo.
echo  Results in: %OUT_BASE%/
echo    sca_mobilenet/     mobilenetv3_base/   mpsnet/
echo    efficientnet_b0/   deit_tiny/          mobilevit_s/
echo    swin_tiny/         rsnet/              fgfnet/
echo    gscl_resnet18/     resnet50/           densenet161/
echo.
echo  Next: Extract EER from each training_metrics.json
echo        and add results table to paper
echo ============================================================

pause
exit /b %FAIL_COUNT%
