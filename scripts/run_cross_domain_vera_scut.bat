@echo off
setlocal
REM ============================================================
REM Cross-Domain Evaluation: VERA <-> SCUT
REM Direction 1: Train VERA -> Test SCUT (10 models)
REM Direction 2: Train SCUT -> Test VERA (5 models)
REM ============================================================

set FAIL_COUNT=0

echo ========================================
echo [A] VERA -> SCUT (10 models)
echo     Load VERA-trained weights, test on entire SCUT
echo ========================================
echo.

python evaluation/cross_domain_vera_scut.py --direction vera_to_scut --all --batch-size 64
if errorlevel 1 (echo [FAILED] VERA->SCUT & set /a FAIL_COUNT+=1) else (echo [OK] VERA->SCUT)
echo.

echo ========================================
echo [B] SCUT -> VERA (5 models)
echo     Load SCUT-trained weights, test on entire VERA
echo ========================================
echo.

python evaluation/cross_domain_vera_scut.py --direction scut_to_vera --all --batch-size 64
if errorlevel 1 (echo [FAILED] SCUT->VERA & set /a FAIL_COUNT+=1) else (echo [OK] SCUT->VERA)
echo.

echo ========================================
echo CROSS-DOMAIN VERA-SCUT SUMMARY
echo   Results: results/results_cross_domain_vera_scut/
echo ========================================
if %FAIL_COUNT%==0 (
    echo ALL COMPLETED SUCCESSFULLY.
) else (
    echo Completed with %FAIL_COUNT% failure(s^).
)
echo ========================================
pause
exit /b %FAIL_COUNT%
