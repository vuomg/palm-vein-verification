@echo off
setlocal

REM ===== Train missing models + run benchmark & gradcam =====
REM   1. EUSIPCO-DenseNet161  [DONE - skip]
REM   2. ResNet-50 (via GSCL --gscl-backbone resnet50)
REM   3. Swin-Tiny (re-train)
REM   4. Inference Benchmark
REM   5. Grad-CAM Visualization
REM ============================================================

set "DATASET_PATH=datasets/final_dataset_openset"
set FAIL_COUNT=0

echo ================================================
echo Run 1/4: ResNet-50 (GSCL framework)
echo ================================================
python train.py ^
  --model gscl ^
  --gscl-backbone resnet50 ^
  --feature-dim 1024 ^
  --epochs 100 ^
  --batch-size 16 ^
  --seed 42 ^
  --dataset "%DATASET_PATH%" ^
  --output-dir results_resnet50

if errorlevel 1 (
  echo [FAILED] ResNet-50
  set /a FAIL_COUNT+=1
) else (
  echo [OK] ResNet-50
)

echo.
echo ================================================
echo Run 2/4: Swin-Tiny (re-train)
echo ================================================
python train.py ^
  --model sca_mobilenet ^
  --sca-backbone swin_tiny ^
  --loss-type adacos_only ^
  --feature-dim 1024 ^
  --epochs 100 ^
  --batch-size 16 ^
  --seed 42 ^
  --dataset "%DATASET_PATH%" ^
  --output-dir results_Swin-Tiny

if errorlevel 1 (
  echo [FAILED] Swin-Tiny
  set /a FAIL_COUNT+=1
) else (
  echo [OK] Swin-Tiny
)

echo.
echo ================================================
echo Run 3/4: Inference Benchmark (all models)
echo ================================================
python evaluation/inference_benchmark.py

if errorlevel 1 (
  echo [FAILED] Inference benchmark
  set /a FAIL_COUNT+=1
) else (
  echo [OK] Inference benchmark
)

echo.
echo ================================================
echo Run 4/4: Grad-CAM Visualization (all models)
echo ================================================
python evaluation/gradcam_visualization.py

if errorlevel 1 (
  echo [FAILED] Grad-CAM visualization
  set /a FAIL_COUNT+=1
) else (
  echo [OK] Grad-CAM visualization
)

echo.
echo ================================================
if %FAIL_COUNT%==0 (
  echo All 4 runs completed successfully.
) else (
  echo Completed with %FAIL_COUNT% failure(s).
)
echo ================================================
exit /b %FAIL_COUNT%
