@echo off
REM ======================================================================
REM ABLATION RUN 3: Baseline + CA only
REM Architecture: MobileNetV3-Small (Variant H) + CA, NO STN, NO SPP (GAP)
REM ======================================================================

echo [INFO] Ablation Run 3: Baseline + CA
echo [CONFIG] Model: mobilenetv3_spp
echo [CONFIG] STN: Disabled ^| CA: Enabled ^| SPP: Disabled (GAP)
echo [CONFIG] Loss: adacos_only (Margin 0.35)
echo [CONFIG] Dropout: 0.3
echo.

python train.py ^
    --model mobilenetv3_spp ^
    --dataset "C:\Research\Research\PalmVein\datasets\final_dataset_openset" ^
    --output-dir results_ablation_ca ^
    --loss-type adacos_only ^
    --feature-dim 1024 ^
    --dropout 0.3 ^
    --batch-size 32 ^
    --epochs 100 ^
    --lr 0.001 ^
    --eval-frequency 5 ^
    --save-best-by val_eer ^
    --no-stn ^
    --no-spp

echo.
echo [DONE] Ablation Run 3 (Baseline + CA) hoan tat!
pause
