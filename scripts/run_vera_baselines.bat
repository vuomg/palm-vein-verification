@echo off
setlocal
REM Train all 9 models on VERA dataset (open-set) - Table II paper
REM Dataset already prepared at VERA_dataset_openset\

set "VERA_OPENSET=datasets/VERA_dataset_openset"
set FAIL_COUNT=0

REM Skip preprocessing - already done (VERA_enhanced + VERA_dataset_openset exist)
if not exist "%VERA_OPENSET%\train" (
    echo [ERROR] Dataset not found: %VERA_OPENSET%\train
    echo Run: python preprocessing/prepare_vera_dataset.py
    echo Then: python preprocessing/split_vera_openset.py
    pause
    exit /b 1
)
echo [OK] Dataset found: %VERA_OPENSET%
echo.

echo ========================================
echo VERA 1/9: MPSNet
echo ========================================
python train.py ^
    --model mpsnet ^
    --dataset "%VERA_OPENSET%" ^
    --output-dir results_vera_mpsnet ^
    --epochs 100 ^
    --batch-size 16 ^
    --database VERA ^
    --eval-frequency 5 ^
    --no-checkpoint ^
    --seed 42
if errorlevel 1 (echo [FAILED] MPSNet & set /a FAIL_COUNT+=1) else (echo [OK] MPSNet)
echo.

echo ========================================
echo VERA 2/9: Modified-DenseNet161
echo ========================================
python train.py ^
    --model eusipco2020 ^
    --eusipco-backbone densenet161 ^
    --dataset "%VERA_OPENSET%" ^
    --output-dir results_vera_densenet161 ^
    --epochs 100 ^
    --batch-size 4 ^
    --database VERA ^
    --eval-frequency 5 ^
    --no-checkpoint ^
    --seed 42
if errorlevel 1 (echo [FAILED] DenseNet161 & set /a FAIL_COUNT+=1) else (echo [OK] DenseNet161)
echo.

echo ========================================
echo VERA 3/9: GSCL
echo ========================================
cd /d "%~dp0..\models\GSCL-PyTorch\vein_feature_learning"
python train_palmvein_fusionaug.py ^
    --trainset "..\..\%VERA_OPENSET%\train" ^
    --testset "..\..\%VERA_OPENSET%\test" ^
    --dataset_name palmvein ^
    --network resnet18 ^
    --loss fusionloss ^
    --max_epoch 100 ^
    --batch_size 64
if errorlevel 1 (echo [FAILED] GSCL & set /a FAIL_COUNT+=1) else (echo [OK] GSCL)
cd /d "%~dp0"
echo.

echo ========================================
echo VERA 4/9: RSNet
echo ========================================
python train.py ^
    --model rsnet ^
    --dataset "%VERA_OPENSET%" ^
    --output-dir results_vera_rsnet ^
    --epochs 100 ^
    --batch-size 32 ^
    --database VERA ^
    --eval-frequency 5 ^
    --no-checkpoint ^
    --seed 42
if errorlevel 1 (echo [FAILED] RSNet & set /a FAIL_COUNT+=1) else (echo [OK] RSNet)
echo.

echo ========================================
echo VERA 5/9: FGFNet
echo ========================================
python train.py ^
    --model fgfnet ^
    --dataset "%VERA_OPENSET%" ^
    --output-dir results_vera_fgfnet ^
    --epochs 100 ^
    --batch-size 4 ^
    --database VERA ^
    --eval-frequency 5 ^
    --no-checkpoint ^
    --seed 42
if errorlevel 1 (echo [FAILED] FGFNet & set /a FAIL_COUNT+=1) else (echo [OK] FGFNet)
echo.

echo ========================================
echo VERA 6/9: ResNet50 (via GSCL framework)
echo ========================================
python train.py ^
    --model gscl ^
    --gscl-backbone resnet50 ^
    --feature-dim 1024 ^
    --dataset "%VERA_OPENSET%" ^
    --output-dir results_vera_resnet50 ^
    --epochs 100 ^
    --batch-size 16 ^
    --database VERA ^
    --eval-frequency 5 ^
    --no-checkpoint ^
    --seed 42
if errorlevel 1 (echo [FAILED] ResNet50 & set /a FAIL_COUNT+=1) else (echo [OK] ResNet50)
echo.

echo ========================================
echo VERA 7/9: MobileNetV3-Base (no STN/CA/SPP)
echo ========================================
python train.py ^
    --model sca_mobilenet ^
    --no-stn --no-ca --no-spp ^
    --dataset "%VERA_OPENSET%" ^
    --output-dir results_vera_mobilenetv3_base ^
    --loss-type adacos_only ^
    --feature-dim 1024 ^
    --dropout 0.3 ^
    --batch-size 16 ^
    --epochs 100 ^
    --lr 0.001 ^
    --database VERA ^
    --eval-frequency 5 ^
    --no-checkpoint ^
    --seed 42
if errorlevel 1 (echo [FAILED] MobileNetV3-Base & set /a FAIL_COUNT+=1) else (echo [OK] MobileNetV3-Base)
echo.

echo ========================================
echo VERA 8/9: EfficientNet-B0
echo ========================================
python train.py ^
    --model sca_mobilenet ^
    --sca-backbone efficientnet_b0 ^
    --dataset "%VERA_OPENSET%" ^
    --output-dir results_vera_efficientnet_b0 ^
    --loss-type adacos_only ^
    --feature-dim 1024 ^
    --dropout 0.3 ^
    --batch-size 16 ^
    --epochs 100 ^
    --lr 0.001 ^
    --database VERA ^
    --eval-frequency 5 ^
    --no-checkpoint ^
    --seed 42
if errorlevel 1 (echo [FAILED] EfficientNet-B0 & set /a FAIL_COUNT+=1) else (echo [OK] EfficientNet-B0)
echo.

echo ========================================
echo VERA 9/9: SCA-MobileNet (proposed)
echo ========================================
python train.py ^
    --model sca_mobilenet ^
    --dataset "%VERA_OPENSET%" ^
    --output-dir results_vera_sca_mobilenet ^
    --loss-type adacos_only ^
    --feature-dim 1024 ^
    --dropout 0.3 ^
    --batch-size 16 ^
    --epochs 100 ^
    --lr 0.001 ^
    --database VERA ^
    --eval-frequency 5 ^
    --no-checkpoint ^
    --seed 42
if errorlevel 1 (echo [FAILED] SCA-MobileNet & set /a FAIL_COUNT+=1) else (echo [OK] SCA-MobileNet)
echo.

:summary
echo ========================================
if %FAIL_COUNT%==0 (
    echo ALL 9 VERA MODELS COMPLETED.
) else (
    echo Completed with %FAIL_COUNT% failure(s^).
)
echo Results: results_vera_*/
echo ========================================
pause
exit /b %FAIL_COUNT%
