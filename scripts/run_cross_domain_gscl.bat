@echo off
REM =====================================================
REM Cross-Domain: GSCL (Train DTS → Test TONGJI)
REM =====================================================

echo "Cross-Domain: GSCL (DTS → TONGJI)"
python evaluation/cross_domain_eval.py --model gscl

pause
