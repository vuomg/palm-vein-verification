@echo off
REM =====================================================
REM Train 6 missing models on TONGJI dataset (Open-Set)
REM =====================================================
REM Paper SCUT/VERA has 12 models, TONGJI only has 6.
REM This trains the 6 missing: ResNet50, MobileNetV3-Base,
REM EfficientNet-B0, Swin-Tiny, DeiT-Tiny, MobileViT-S
REM =====================================================
REM Dataset: TONGJI_dataset_openset (840 train / 360 test)
REM Estimated GPU time: ~12-15h total
REM =====================================================

echo =====================================================
echo [1/6] Training MobileNetV3-Base on TONGJI...
echo =====================================================
python train.py --model sca_mobilenet --no-stn --no-ca --no-spp ^
    --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_mobilenetv3_base ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database TJ_PV --eval-frequency 5 --seed 42
if %ERRORLEVEL% neq 0 echo [FAILED] MobileNetV3-Base && goto model2
echo [DONE] MobileNetV3-Base

:model2
echo =====================================================
echo [2/6] Training EfficientNet-B0 on TONGJI...
echo =====================================================
python train.py --model sca_mobilenet --sca-backbone efficientnet_b0 ^
    --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_efficientnet_b0 ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database TJ_PV --eval-frequency 5 --seed 42
if %ERRORLEVEL% neq 0 echo [FAILED] EfficientNet-B0 && goto model3
echo [DONE] EfficientNet-B0

:model3
echo =====================================================
echo [3/6] Training Swin-Tiny on TONGJI...
echo =====================================================
python train.py --model sca_mobilenet --sca-backbone swin_tiny ^
    --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_swin_tiny ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database TJ_PV --eval-frequency 5 --seed 42
if %ERRORLEVEL% neq 0 echo [FAILED] Swin-Tiny && goto model4
echo [DONE] Swin-Tiny

:model4
echo =====================================================
echo [4/6] Training DeiT-Tiny on TONGJI...
echo =====================================================
python train.py --model sca_mobilenet --sca-backbone deit_tiny ^
    --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_deit_tiny ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database TJ_PV --eval-frequency 5 --seed 42
if %ERRORLEVEL% neq 0 echo [FAILED] DeiT-Tiny && goto model5
echo [DONE] DeiT-Tiny

:model5
echo =====================================================
echo [5/6] Training MobileViT-S on TONGJI...
echo =====================================================
python train.py --model sca_mobilenet --sca-backbone mobilevit_s ^
    --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_mobilevit_s ^
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
    --batch-size 16 --epochs 100 --lr 0.001 ^
    --database TJ_PV --eval-frequency 5 --seed 42
if %ERRORLEVEL% neq 0 echo [FAILED] MobileViT-S && goto model6
echo [DONE] MobileViT-S

:model6
echo =====================================================
echo [6/6] Training ResNet50 on TONGJI (GSCL framework)...
echo =====================================================
REM ResNet50 uses GSCL-PyTorch framework with separate conda env
REM Uncomment and adjust if running in GSCL conda env:
REM conda activate gscl
REM cd models/GSCL-PyTorch/vein_feature_learning
REM python train_palmvein_fusionaug.py ^
REM     --trainset datasets/TONGJI_dataset_openset/train ^
REM     --testset datasets/TONGJI_dataset_openset/test ^
REM     --dataset_name tongji_resnet50 ^
REM     --network resnet50 --loss fusionloss ^
REM     --max_epoch 100 --p 16 --k 4 --lr 0.01 --eval_freq 5 --seed 42
echo [NOTE] ResNet50 requires GSCL conda env - run manually:
echo   conda activate gscl
echo   cd models/GSCL-PyTorch/vein_feature_learning
echo   python train_palmvein_fusionaug.py --trainset datasets/TONGJI_dataset_openset/train --testset datasets/TONGJI_dataset_openset/test --dataset_name tongji_resnet50 --network resnet50 --loss fusionloss --max_epoch 100 --p 16 --k 4 --lr 0.01 --eval_freq 5 --seed 42

echo =====================================================
echo All TONGJI training complete!
echo Results saved to results_tongji_*/
echo =====================================================
pause
