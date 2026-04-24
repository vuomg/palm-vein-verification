@echo off
setlocal
REM Train all 9 models on VERA dataset WITHOUT CLAHE preprocessing
REM Compare with run_vera_baselines.bat (with CLAHE) to measure preprocessing impact

set "VERA_OPENSET=datasets/VERA_noclahe_openset"
set "GSCL_DIR=%~dp0..\models\GSCL-PyTorch\vein_feature_learning"
set FAIL_COUNT=0

if not exist "%VERA_OPENSET%\train" (
    echo [ERROR] Dataset not found: %VERA_OPENSET%\train
    echo Run: python preprocessing/prepare_vera_noclahe.py
    echo Then: python preprocessing/split_vera_openset.py --input VERA_noclahe --output VERA_noclahe_openset
    pause & exit /b 1
)
echo [OK] Dataset: %VERA_OPENSET%
echo.

echo ========================================
echo VERA-NoCLAHE 1/9: MPSNet
echo ========================================
python train.py ^
    --model mpsnet ^
    --dataset "%VERA_OPENSET%" ^
    --output-dir results_vera_noclahe_mpsnet ^
    --epochs 100 --batch-size 16 ^
    --database VERA --eval-frequency 5 ^
    --no-checkpoint --seed 42
if errorlevel 1 (echo [FAILED] MPSNet & set /a FAIL_COUNT+=1) else (echo [OK] MPSNet)
echo.

echo ========================================
echo VERA-NoCLAHE 2/9: Modified-DenseNet161
echo ========================================
python train.py ^
    --model eusipco2020 --eusipco-backbone densenet161 ^
    --dataset "%VERA_OPENSET%" ^
    --output-dir results_vera_noclahe_densenet161 ^
    --epochs 100 --batch-size 4 ^
    --database VERA --eval-frequency 5 ^
    --no-checkpoint --seed 42
if errorlevel 1 (echo [FAILED] DenseNet161 & set /a FAIL_COUNT+=1) else (echo [OK] DenseNet161)
echo.

echo ========================================
echo VERA-NoCLAHE 3/9: GSCL (ResNet18)
echo ========================================
cd /d "%GSCL_DIR%"
python train_palmvein_fusionaug.py ^
    --trainset "%~dp0%VERA_OPENSET%\train" ^
    --testset  "%~dp0%VERA_OPENSET%\test" ^
    --dataset_name vera_noclahe_resnet18 ^
    --network resnet18 --loss fusionloss ^
    --max_epoch 100 --p 16 --k 4 ^
    --lr 0.01 --eval_freq 5 --seed 42
if errorlevel 1 (
    echo [FAILED] GSCL & set /a FAIL_COUNT+=1
) else (
    echo [OK] GSCL
    if not exist "%~dp0results_vera_noclahe_gscl" mkdir "%~dp0results_vera_noclahe_gscl"
    copy /Y "results\vera_noclahe_resnet18_resnet18\training_metrics.json" "%~dp0results_vera_noclahe_gscl\training_metrics.json"
)
cd /d "%~dp0"
echo.

echo ========================================
echo VERA-NoCLAHE 4/9: RSNet
echo ========================================
python train.py ^
    --model rsnet ^
    --dataset "%VERA_OPENSET%" ^
    --output-dir results_vera_noclahe_rsnet ^
    --epochs 100 --batch-size 32 ^
    --database VERA --eval-frequency 5 ^
    --no-checkpoint --seed 42
if errorlevel 1 (echo [FAILED] RSNet & set /a FAIL_COUNT+=1) else (echo [OK] RSNet)
echo.

echo ========================================
echo VERA-NoCLAHE 5/9: FGFNet
echo ========================================
python train.py ^
    --model fgfnet ^
    --dataset "%VERA_OPENSET%" ^
    --output-dir results_vera_noclahe_fgfnet ^
    --epochs 100 --batch-size 4 ^
    --database VERA --eval-frequency 5 ^
    --no-checkpoint --seed 42
if errorlevel 1 (echo [FAILED] FGFNet & set /a FAIL_COUNT+=1) else (echo [OK] FGFNet)
echo.

echo ========================================
echo VERA-NoCLAHE 6/9: ResNet50
echo ========================================
cd /d "%GSCL_DIR%"
python train_palmvein_fusionaug.py ^
    --trainset "%~dp0%VERA_OPENSET%\train" ^
    --testset  "%~dp0%VERA_OPENSET%\test" ^
    --dataset_name vera_noclahe_resnet50 ^
    --network resnet50 --loss fusionloss ^
    --max_epoch 100 --p 16 --k 4 ^
    --lr 0.01 --eval_freq 5 --seed 42
if errorlevel 1 (
    echo [FAILED] ResNet50 & set /a FAIL_COUNT+=1
) else (
    echo [OK] ResNet50
    if not exist "%~dp0results_vera_noclahe_resnet50" mkdir "%~dp0results_vera_noclahe_resnet50"
    copy /Y "results\vera_noclahe_resnet50_resnet50\training_metrics.json" "%~dp0results_vera_noclahe_resnet50\training_metrics.json"
)
cd /d "%~dp0"
echo.

echo ========================================
echo VERA-NoCLAHE 7/9: MobileNetV3-Base
echo ========================================
python train.py ^
    --model sca_mobilenet --no-stn --no-ca --no-spp ^
    --dataset "%VERA_OPENSET%" ^
    --output-dir results_vera_noclahe_mobilenetv3_base ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database VERA --eval-frequency 5 ^
    --no-checkpoint --seed 42
if errorlevel 1 (echo [FAILED] MobileNetV3-Base & set /a FAIL_COUNT+=1) else (echo [OK] MobileNetV3-Base)
echo.

echo ========================================
echo VERA-NoCLAHE 8/9: EfficientNet-B0
echo ========================================
python train.py ^
    --model sca_mobilenet --sca-backbone efficientnet_b0 ^
    --dataset "%VERA_OPENSET%" ^
    --output-dir results_vera_noclahe_efficientnet_b0 ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database VERA --eval-frequency 5 ^
    --no-checkpoint --seed 42
if errorlevel 1 (echo [FAILED] EfficientNet-B0 & set /a FAIL_COUNT+=1) else (echo [OK] EfficientNet-B0)
echo.

echo ========================================
echo VERA-NoCLAHE 9/9: SCA-MobileNet (proposed)
echo ========================================
python train.py ^
    --model sca_mobilenet ^
    --dataset "%VERA_OPENSET%" ^
    --output-dir results_vera_noclahe_sca_mobilenet ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database VERA --eval-frequency 5 ^
    --no-checkpoint --seed 42
if errorlevel 1 (echo [FAILED] SCA-MobileNet & set /a FAIL_COUNT+=1) else (echo [OK] SCA-MobileNet)
echo.

:summary
echo ========================================
if %FAIL_COUNT%==0 (
    echo ALL 12 VERA-NoCLAHE MODELS COMPLETED.
) else (
    echo Completed with %FAIL_COUNT% failure(s^).
)
echo Results: results_vera_noclahe_*/
echo ========================================
pause
exit /b %FAIL_COUNT%
