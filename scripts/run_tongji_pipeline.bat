@echo off
REM Train SCA-MobileNet on TONGJI dataset
REM Step 1: Prepare and enhance
python preprocessing/prepare_tongji_dataset.py

REM Step 2: Split train/test
python preprocessing/split_tongji_openset.py

REM Step 3: Train SCA-MobileNet (full 100 epochs)
python train.py --model mobilenetv3_spp --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji --epochs 100 --batch-size 16 --database TJ_PV --eval-frequency 5

pause
