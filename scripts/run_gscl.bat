@echo off
REM =====================================================
REM Train GSCL Model on Palm Vein dataset (Open-Set)
REM =====================================================
REM Uses FusionLoss (CosFace + Triplet) with ResNet18
REM Evaluates every 5 epochs to match other models
REM =====================================================

echo "Training GSCL (FusionAug) on Palm Vein Open-Set..."
cd /d c:\Research\Research\PalmVein\models\GSCL-PyTorch\vein_feature_learning

python train_palmvein_fusionaug.py ^
    --trainset "c:\Research\Research\PalmVein\datasets\final_dataset_openset\train" ^
    --testset "c:\Research\Research\PalmVein\datasets\final_dataset_openset\test" ^
    --dataset_name palmvein ^
    --network resnet18 ^
    --loss fusionloss ^
    --max_epoch 100 ^
    --lr 0.01 ^
    --p 16 ^
    --k 4 ^
    --s 30.0 ^
    --m 0.2 ^
    --hard_margin 0.2 ^
    --w_cls 1.0 ^
    --w_metric 4.0 ^
    --eval_freq 5 ^
    --seed 42

pause
