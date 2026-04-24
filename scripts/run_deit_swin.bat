@echo off
setlocal

REM ===== Configure dataset path before running =====
set "DATASET_PATH=datasets/final_dataset_openset"

echo ========================================
echo Run 1/2: SCA-MobileNet with DeiT-Tiny
echo ========================================
python train.py ^
  --model sca_mobilenet ^
  --sca-backbone deit_tiny ^
  --loss-type adacos_only ^
  --feature-dim 1024 ^
  --dataset "%DATASET_PATH%"

if errorlevel 1 (
  echo.
  echo DeiT-Tiny run failed. Stopping batch.
  exit /b 1
)

echo.
echo ========================================
echo Run 2/2: SCA-MobileNet with Swin-Tiny
echo ========================================
python train.py ^
  --model sca_mobilenet ^
  --sca-backbone swin_tiny ^
  --loss-type adacos_only ^
  --feature-dim 1024 ^
  --dataset "%DATASET_PATH%"

if errorlevel 1 (
  echo.
  echo Swin-Tiny run failed.
  exit /b 1
)

echo.
echo Both runs completed successfully.
exit /b 0
