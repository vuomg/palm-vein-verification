@echo off
REM ======================================================================
REM ABLATION RUN 2: Baseline + STN only
REM Architecture: MobileNetV3-Small (Variant H) + STN, NO CA, NO SPP (GAP)
REM ======================================================================

echo [INFO] Ablation Run 2: Baseline + STN
echo [CONFIG] Model: mobilenetv3_spp
echo [CONFIG] STN: Enabled ^| CA: Disabled ^| SPP: Disabled (GAP)
echo [CONFIG] Loss: adacos_only (Margin 0.35)
echo [CONFIG] Dropout: 0.3
echo.

python train.py ^
    --model mobilenetv3_spp ^
    --dataset "C:\Research\Research\PalmVein\datasets\final_dataset_openset" ^
    --output-dir results_ablation_stn ^
    --loss-type adacos_only ^
    --feature-dim 1024 ^
    --dropout 0.3 ^
    --batch-size 32 ^
    --epochs 100 ^
    --lr 0.001 ^
    --eval-frequency 5 ^
    --save-best-by val_eer ^
    --no-ca ^
    --no-spp

echo.
echo [DONE] Ablation Run 2 (Baseline + STN) hoan tat!
pause
