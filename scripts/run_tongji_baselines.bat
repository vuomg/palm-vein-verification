@echo off
REM =====================================================
REM Train Baseline SOTA Models on TONGJI dataset (Open-Set)
REM =====================================================
REM This script runs training for other state-of-the-art models
REM on the same TONGJI_dataset_openset for fair comparison.
REM =====================================================

echo "Training MPSNet on TONGJI..."
python train.py --model mpsnet --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_mpsnet --epochs 100 --batch-size 16 --database TJ_PV --eval-frequency 5

echo "Training GSCL on TONGJI..."
python train.py --model gscl --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_gscl --epochs 100 --batch-size 16 --database TJ_PV --eval-frequency 5

echo "Training RSNet on TONGJI..."
python train.py --model rsnet --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_rsnet --epochs 100 --batch-size 16 --database TJ_PV --eval-frequency 5

pause
