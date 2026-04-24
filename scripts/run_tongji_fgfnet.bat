@echo off
REM =====================================================
REM Train FGFNet Model on TONGJI dataset (Open-Set)
REM =====================================================
REM This script runs FGFNet specifically.
REM Batch size is reduced (e.g., 4 or 8) because FGFNet
REM requires significantly more GPU memory.
REM =====================================================

echo "Training FGFNet on TONGJI (Reduced Batch Size)..."
python train.py --model fgfnet --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_fgfnet --epochs 100 --batch-size 4 --database TJ_PV --eval-frequency 5

pause
