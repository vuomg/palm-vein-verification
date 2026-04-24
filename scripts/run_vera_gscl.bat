@echo off
setlocal
REM Run GSCL (ResNet18) and ResNet50 on VERA dataset
REM Output: models\GSCL-PyTorch\vein_feature_learning\results\vera_*
REM Then copy training_metrics.json to results_vera_gscl_* for unified reporting

set "ROOT=C:\Research\Research\PalmVein"
set "GSCL_DIR=%ROOT%\models\GSCL-PyTorch\vein_feature_learning"
set "TRAINSET=%ROOT%\datasets\VERA_dataset_openset\train"
set "TESTSET=%ROOT%\datasets\VERA_dataset_openset\test"
set FAIL_COUNT=0

if not exist "%TRAINSET%" (
    echo [ERROR] Dataset not found: %TRAINSET%
    pause & exit /b 1
)

cd /d "%GSCL_DIR%"

echo ========================================
echo VERA GSCL 1/2: ResNet18 (GSCL baseline)
echo         seed=42, fusionloss, 100 epochs
echo ========================================
python train_palmvein_fusionaug.py ^
    --trainset "%TRAINSET%" ^
    --testset "%TESTSET%" ^
    --dataset_name vera_resnet18 ^
    --network resnet18 ^
    --loss fusionloss ^
    --max_epoch 100 ^
    --p 16 --k 4 ^
    --lr 0.01 ^
    --eval_freq 5 ^
    --seed 42
if errorlevel 1 (
    echo [FAILED] GSCL ResNet18
    set /a FAIL_COUNT+=1
) else (
    echo [OK] GSCL ResNet18
    REM Copy metrics to main results dir for unified reporting
    if not exist "%ROOT%\results_vera_gscl_resnet18" mkdir "%ROOT%\results_vera_gscl_resnet18"
    copy /Y "results\vera_resnet18\training_metrics.json" "%ROOT%\results_vera_gscl_resnet18\training_metrics.json"
    echo [OK] Metrics copied to results_vera_gscl_resnet18\
)
echo.

echo ========================================
echo VERA GSCL 2/2: ResNet50 (baseline)
echo         seed=42, fusionloss, 100 epochs
echo ========================================
python train_palmvein_fusionaug.py ^
    --trainset "%TRAINSET%" ^
    --testset "%TESTSET%" ^
    --dataset_name vera_resnet50 ^
    --network resnet50 ^
    --loss fusionloss ^
    --max_epoch 100 ^
    --p 16 --k 4 ^
    --lr 0.01 ^
    --eval_freq 5 ^
    --seed 42
if errorlevel 1 (
    echo [FAILED] ResNet50
    set /a FAIL_COUNT+=1
) else (
    echo [OK] ResNet50
    if not exist "%ROOT%\results_vera_resnet50" mkdir "%ROOT%\results_vera_resnet50"
    copy /Y "results\vera_resnet50\training_metrics.json" "%ROOT%\results_vera_resnet50\training_metrics.json"
    echo [OK] Metrics copied to results_vera_resnet50\
)
echo.

cd /d "%ROOT%"

echo ========================================
if %FAIL_COUNT%==0 (
    echo GSCL VERA: ALL COMPLETED.
) else (
    echo GSCL VERA: %FAIL_COUNT% failure(s^).
)
echo ========================================
pause
exit /b %FAIL_COUNT%
