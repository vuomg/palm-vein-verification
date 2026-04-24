@echo off
REM =====================================================
REM Train Modified-DenseNet161 on TONGJI dataset (Open-Set)
REM =====================================================
REM Model:   Modified-DenseNet161 (EUSIPCO 2020)
REM Dataset: TONGJI_dataset_openset (1200 identities, ROI pre-cropped)
REM Protocol: Open-Set 70/30 identity-level split
REM =====================================================

echo "Training Modified-DenseNet161 on TONGJI..."
python train.py --model eusipco2020 --eusipco-backbone densenet161 --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_densenet161 --epochs 100 --batch-size 4 --database TJ_PV --eval-frequency 5

pause
