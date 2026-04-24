@echo off
REM ======================================================================
REM SCRIPT TRAIN SCA-MobileNet v2 (Self-Contained Backbone)
REM Architecture: STN + Custom Backbone (Variant H) + SCIB(CA+SPP)
REM Loss: AdaCos + Geometric Regularization
REM ======================================================================

echo [INFO] Training SCA-MobileNet v2 (Self-Contained)...
echo [CONFIG] Model: sca_mobilenet
echo [CONFIG] Loss: adacos_only (Margin 0.35) + GeoReg
echo [CONFIG] Backbone: Self-contained (no torchvision dependency)
echo.

python train.py ^
    --model sca_mobilenet ^
    --dataset "C:\Research\Research\PalmVein\datasets\final_dataset_openset" ^
    --output-dir results_sca ^
    --loss-type adacos_only ^
    --feature-dim 1024 ^
    --dropout 0.3 ^
    --batch-size 32 ^
    --epochs 100 ^
    --lr 0.001 ^
    --eval-frequency 5 ^
    --save-best-by val_eer

echo.
echo [DONE] Training complete!
pause
