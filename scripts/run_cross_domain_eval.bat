@echo off
REM =====================================================
REM Cross-Domain Evaluation: Train DTS → Test TONGJI
REM All models (MPSNet, DenseNet161, GSCL, RSNet, FGFNet, SCA-MobileNet)
REM =====================================================

echo ========================================
echo Cross-Domain: MPSNet (DTS → TONGJI)
echo ========================================
python evaluation/cross_domain_eval.py --model mpsnet

echo.
echo ========================================
echo Cross-Domain: Modified-DenseNet161 (DTS → TONGJI)
echo ========================================
python evaluation/cross_domain_eval.py --model eusipco2020

echo.
echo ========================================
echo Cross-Domain: GSCL (DTS → TONGJI)
echo ========================================
python evaluation/cross_domain_eval.py --model gscl

echo.
echo ========================================
echo Cross-Domain: RSNet (DTS → TONGJI)
echo ========================================
python evaluation/cross_domain_eval.py --model rsnet

echo.
echo ========================================
echo Cross-Domain: FGFNet (DTS → TONGJI)
echo ========================================
python evaluation/cross_domain_eval.py --model fgfnet --batch-size 16

echo.
echo ========================================
echo Cross-Domain: SCA-MobileNet (DTS → TONGJI)
echo ========================================
python evaluation/cross_domain_eval.py --model sca_mobilenet

echo.
echo ========================================
echo ALL DONE! Results in results_cross_domain/
echo ========================================

pause
