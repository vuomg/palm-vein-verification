@echo off
REM ======================================================================
REM ABLATION RUN 4: Baseline + SPP only
REM Architecture: MobileNetV3-Small (Variant H) + SPP [1,2,4], NO STN, NO CA
REM ======================================================================

echo [INFO] Ablation Run 4: Baseline + SPP
echo [CONFIG] Model: mobilenetv3_spp
echo [CONFIG] STN: Disabled ^| CA: Disabled ^| SPP: Enabled [1,2,4]
echo [CONFIG] Loss: adacos_only (Margin 0.35)
echo [CONFIG] Dropout: 0.3
echo.

python train.py ^
    --model mobilenetv3_spp ^
    --dataset "C:\Research\Research\PalmVein\datasets\final_dataset_openset" ^
    --output-dir results_ablation_spp ^
    --loss-type adacos_only ^
    --feature-dim 1024 ^
    --dropout 0.3 ^
    --batch-size 32 ^
    --epochs 100 ^
    --lr 0.001 ^
    --eval-frequency 5 ^
    --save-best-by val_eer ^
    --no-stn ^
    --no-ca

echo.
echo [DONE] Ablation Run 4 (Baseline + SPP) hoan tat!
pause
