@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

REM ================================================================
REM  RUN ALL - Train TONGJI missing + Cross-Domain VERA<->SCUT
REM  Output hien tren man hinh, loi thi tiep tuc chay model ke tiep
REM ================================================================

set TOTAL=0
set PASS=0
set FAIL=0
set FAIL_LIST=

echo ========================================================
echo   RUN ALL PAPER EXPERIMENTS
echo   Started: %date% %time%
echo ========================================================

REM ================================================================
REM  PART A: Train 5 TONGJI missing models
REM ================================================================
echo.
echo ========================================================
echo   PART A: TRAIN TONGJI MISSING MODELS [5 models]
echo ========================================================

REM --- A1: MobileNetV3-Base ---
set /a TOTAL+=1
echo.
echo [!TOTAL!/29] ===== TONGJI-MobileNetV3-Base ===== %time%
python train.py --model sca_mobilenet --no-stn --no-ca --no-spp --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_mobilenetv3_base --loss-type adacos_only --feature-dim 1024 --dropout 0.3 --batch-size 16 --epochs 100 --lr 0.001 --database TJ_PV --eval-frequency 5 --seed 42
if !ERRORLEVEL! neq 0 (
    set /a FAIL+=1
    set "FAIL_LIST=!FAIL_LIST! TONGJI-MobileNetV3-Base"
    echo [!TOTAL!/29] FAILED TONGJI-MobileNetV3-Base
) else (
    set /a PASS+=1
    echo [!TOTAL!/29] PASS TONGJI-MobileNetV3-Base
)

REM --- A2: EfficientNet-B0 ---
set /a TOTAL+=1
echo.
echo [!TOTAL!/29] ===== TONGJI-EfficientNet-B0 ===== %time%
python train.py --model sca_mobilenet --sca-backbone efficientnet_b0 --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_efficientnet_b0 --loss-type adacos_only --feature-dim 1024 --dropout 0.3 --batch-size 16 --epochs 100 --lr 0.001 --database TJ_PV --eval-frequency 5 --seed 42
if !ERRORLEVEL! neq 0 (
    set /a FAIL+=1
    set "FAIL_LIST=!FAIL_LIST! TONGJI-EfficientNet-B0"
    echo [!TOTAL!/29] FAILED TONGJI-EfficientNet-B0
) else (
    set /a PASS+=1
    echo [!TOTAL!/29] PASS TONGJI-EfficientNet-B0
)

REM --- A3: Swin-Tiny ---
set /a TOTAL+=1
echo.
echo [!TOTAL!/29] ===== TONGJI-Swin-Tiny ===== %time%
python train.py --model sca_mobilenet --sca-backbone swin_tiny --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_swin_tiny --loss-type adacos_only --feature-dim 1024 --dropout 0.3 --batch-size 16 --epochs 100 --lr 0.001 --database TJ_PV --eval-frequency 5 --seed 42
if !ERRORLEVEL! neq 0 (
    set /a FAIL+=1
    set "FAIL_LIST=!FAIL_LIST! TONGJI-Swin-Tiny"
    echo [!TOTAL!/29] FAILED TONGJI-Swin-Tiny
) else (
    set /a PASS+=1
    echo [!TOTAL!/29] PASS TONGJI-Swin-Tiny
)

REM --- A4: DeiT-Tiny ---
set /a TOTAL+=1
echo.
echo [!TOTAL!/29] ===== TONGJI-DeiT-Tiny ===== %time%
python train.py --model sca_mobilenet --sca-backbone deit_tiny --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_deit_tiny --loss-type adacos_only --feature-dim 1024 --dropout 0.3 --batch-size 16 --epochs 100 --lr 0.001 --database TJ_PV --eval-frequency 5 --seed 42
if !ERRORLEVEL! neq 0 (
    set /a FAIL+=1
    set "FAIL_LIST=!FAIL_LIST! TONGJI-DeiT-Tiny"
    echo [!TOTAL!/29] FAILED TONGJI-DeiT-Tiny
) else (
    set /a PASS+=1
    echo [!TOTAL!/29] PASS TONGJI-DeiT-Tiny
)

REM --- A5: MobileViT-S ---
set /a TOTAL+=1
echo.
echo [!TOTAL!/29] ===== TONGJI-MobileViT-S ===== %time%
python train.py --model sca_mobilenet --sca-backbone mobilevit_s --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_mobilevit_s --loss-type adacos_only --feature-dim 1024 --dropout 0.3 --batch-size 16 --epochs 100 --lr 0.001 --database TJ_PV --eval-frequency 5 --seed 42
if !ERRORLEVEL! neq 0 (
    set /a FAIL+=1
    set "FAIL_LIST=!FAIL_LIST! TONGJI-MobileViT-S"
    echo [!TOTAL!/29] FAILED TONGJI-MobileViT-S
) else (
    set /a PASS+=1
    echo [!TOTAL!/29] PASS TONGJI-MobileViT-S
)

echo.
echo [SKIP] TONGJI-ResNet50 - can conda env gscl, chay rieng

REM ================================================================
REM  PART B: Cross-Domain VERA -> SCUT (12 models)
REM ================================================================
echo.
echo ========================================================
echo   PART B: CROSS-DOMAIN VERA -^> SCUT [12 models]
echo ========================================================

for %%m in (sca_mobilenet mpsnet eusipco2020 rsnet fgfnet mobilenetv3_base efficientnet_b0 deit_tiny swin_tiny mobilevit_s gscl resnet50) do (
    set /a TOTAL+=1
    echo.
    echo [!TOTAL!/29] ===== VERA-to-SCUT-%%m ===== !time!
    python evaluation/cross_domain_vera_scut.py --direction vera_to_scut --model %%m --batch-size 64
    if !ERRORLEVEL! neq 0 (
        set /a FAIL+=1
        set "FAIL_LIST=!FAIL_LIST! VERA-to-SCUT-%%m"
        echo [!TOTAL!/29] FAILED VERA-to-SCUT-%%m
    ) else (
        set /a PASS+=1
        echo [!TOTAL!/29] PASS VERA-to-SCUT-%%m
    )
)

REM ================================================================
REM  PART C: Cross-Domain SCUT -> VERA (12 models)
REM ================================================================
echo.
echo ========================================================
echo   PART C: CROSS-DOMAIN SCUT -^> VERA [12 models]
echo ========================================================

for %%m in (sca_mobilenet mpsnet eusipco2020 rsnet fgfnet mobilenetv3_base efficientnet_b0 deit_tiny swin_tiny mobilevit_s gscl resnet50) do (
    set /a TOTAL+=1
    echo.
    echo [!TOTAL!/29] ===== SCUT-to-VERA-%%m ===== !time!
    python evaluation/cross_domain_vera_scut.py --direction scut_to_vera --model %%m --batch-size 64
    if !ERRORLEVEL! neq 0 (
        set /a FAIL+=1
        set "FAIL_LIST=!FAIL_LIST! SCUT-to-VERA-%%m"
        echo [!TOTAL!/29] FAILED SCUT-to-VERA-%%m
    ) else (
        set /a PASS+=1
        echo [!TOTAL!/29] PASS SCUT-to-VERA-%%m
    )
)

REM ================================================================
REM  FINAL REPORT
REM ================================================================
echo.
echo ========================================================
echo   FINAL REPORT  -  %date% %time%
echo ========================================================
echo   Total : !TOTAL!
echo   Pass  : !PASS!
echo   Fail  : !FAIL!
echo ========================================================
if !FAIL! gtr 0 (
    echo   FAILED: !FAIL_LIST!
)
echo ========================================================
