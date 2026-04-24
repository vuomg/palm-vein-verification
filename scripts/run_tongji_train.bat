@echo off
REM =====================================================
REM Train SCA-MobileNet on TONGJI dataset (Open-Set)
REM Enhancement and Split already completed.
REM =====================================================
REM Dataset: TONGJI_dataset_openset (840 train / 360 test users)
REM Model:   SCA-MobileNet (using --model sca_mobilenet with STN + CA + SPP)
REM =====================================================

python train.py --model sca_mobilenet --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji --epochs 100 --batch-size 16 --database TJ_PV --eval-frequency 5

pause
