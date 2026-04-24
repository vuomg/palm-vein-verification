@echo off
setlocal
REM Train remaining 6 models on SCUT dataset
REM Already completed (on another machine): MPSNet, DenseNet161, FGFNet, RSNet, MobileNetV3-Base, SCA-MobileNet
REM This script runs: GSCL, ResNet50, EfficientNet-B0, Swin-Tiny, DeiT-Tiny, MobileViT-S

set "SCUT_OPENSET=datasets/SCUT_dataset_openset"
set "GSCL_DIR=%~dp0..\models\GSCL-PyTorch\vein_feature_learning"
set FAIL_COUNT=0

if not exist "%SCUT_OPENSET%\train" (
    echo [ERROR] Dataset not found: %SCUT_OPENSET%\train
    echo Run: python preprocessing/prepare_scut_dataset.py
    echo Then: python preprocessing/split_scut_openset.py
    pause & exit /b 1
)
echo [OK] Dataset: %SCUT_OPENSET%
echo.

echo ========================================
echo SCUT 1/6: GSCL (ResNet18)
echo ========================================
cd /d "%GSCL_DIR%"
python train_palmvein_fusionaug.py ^
    --trainset "%~dp0%SCUT_OPENSET%\train" ^
    --testset  "%~dp0%SCUT_OPENSET%\test" ^
    --dataset_name scut_resnet18 ^
    --network resnet18 --loss fusionloss ^
    --max_epoch 100 --p 16 --k 4 ^
    --lr 0.01 --eval_freq 5 --seed 42
if errorlevel 1 (
    echo [FAILED] GSCL & set /a FAIL_COUNT+=1
) else (
    echo [OK] GSCL
    if not exist "%~dp0results_scut_gscl" mkdir "%~dp0results_scut_gscl"
    copy /Y "results\scut_resnet18_resnet18\training_metrics.json" "%~dp0results_scut_gscl\training_metrics.json"
)
cd /d "%~dp0"
echo.

echo ========================================
echo SCUT 2/6: ResNet50
echo ========================================
cd /d "%GSCL_DIR%"
python train_palmvein_fusionaug.py ^
    --trainset "%~dp0%SCUT_OPENSET%\train" ^
    --testset  "%~dp0%SCUT_OPENSET%\test" ^
    --dataset_name scut_resnet50 ^
    --network resnet50 --loss fusionloss ^
    --max_epoch 100 --p 16 --k 4 ^
    --lr 0.01 --eval_freq 5 --seed 42
if errorlevel 1 (
    echo [FAILED] ResNet50 & set /a FAIL_COUNT+=1
) else (
    echo [OK] ResNet50
    if not exist "%~dp0results_scut_resnet50" mkdir "%~dp0results_scut_resnet50"
    copy /Y "results\scut_resnet50_resnet50\training_metrics.json" "%~dp0results_scut_resnet50\training_metrics.json"
)
cd /d "%~dp0"
echo.

echo ========================================
echo SCUT 3/6: EfficientNet-B0
echo ========================================
python train.py ^
    --model sca_mobilenet --sca-backbone efficientnet_b0 ^
    --dataset "%SCUT_OPENSET%" ^
    --output-dir results_scut_efficientnet_b0 ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database SCUT --eval-frequency 5 ^
    --no-checkpoint --seed 42
if errorlevel 1 (echo [FAILED] EfficientNet-B0 & set /a FAIL_COUNT+=1) else (echo [OK] EfficientNet-B0)
echo.

echo ========================================
echo SCUT 4/6: Swin-Tiny
echo ========================================
python train.py ^
    --model sca_mobilenet --sca-backbone swin_tiny ^
    --dataset "%SCUT_OPENSET%" ^
    --output-dir results_scut_swin_tiny ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database SCUT --eval-frequency 5 ^
    --no-checkpoint --seed 42
if errorlevel 1 (echo [FAILED] Swin-Tiny & set /a FAIL_COUNT+=1) else (echo [OK] Swin-Tiny)
echo.

echo ========================================
echo SCUT 5/6: DeiT-Tiny
echo ========================================
python train.py ^
    --model sca_mobilenet --sca-backbone deit_tiny ^
    --dataset "%SCUT_OPENSET%" ^
    --output-dir results_scut_deit_tiny ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database SCUT --eval-frequency 5 ^
    --no-checkpoint --seed 42
if errorlevel 1 (echo [FAILED] DeiT-Tiny & set /a FAIL_COUNT+=1) else (echo [OK] DeiT-Tiny)
echo.

echo ========================================
echo SCUT 6/6: MobileViT-S
echo ========================================
python train.py ^
    --model sca_mobilenet --sca-backbone mobilevit_s ^
    --dataset "%SCUT_OPENSET%" ^
    --output-dir results_scut_mobilevit_s ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database SCUT --eval-frequency 5 ^
    --no-checkpoint --seed 42
if errorlevel 1 (echo [FAILED] MobileViT-S & set /a FAIL_COUNT+=1) else (echo [OK] MobileViT-S)
echo.

:summary
echo ========================================
if %FAIL_COUNT%==0 (
    echo ALL 6 REMAINING SCUT MODELS COMPLETED.
) else (
    echo Completed with %FAIL_COUNT% failure(s^).
)
echo Results: results_scut_*/
echo ========================================
pause
exit /b %FAIL_COUNT%
