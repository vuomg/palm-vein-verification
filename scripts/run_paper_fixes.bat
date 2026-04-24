@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

REM ============================================================
REM Paper Fixes: A (multi-seed), F (benchmark), G (statistics)
REM Estimated time: ~3-4 hours on RTX 4050 Laptop
REM ============================================================

set "SCUT_DATASET=datasets/SCUT_dataset_openset"
set "INTERNAL_DATASET=datasets/FYODB_dataset_openset"

echo ============================================================
echo  PAPER FIXES - Tasks A, F, G
echo  Started: %date% %time%
echo ============================================================

REM Check dataset exists
if not exist "%SCUT_DATASET%\train" (
    echo [ERROR] SCUT dataset not found: %SCUT_DATASET%\train
    echo Run: python preprocessing/prepare_scut_dataset.py
    pause
    exit /b 1
)

REM ============================================================
REM TASK A: Multi-seed SCA-MobileNet on SCUT (5 seeds)
REM Estimate: ~25 min x 5 = ~2-2.5 hours
REM ============================================================
echo.
echo ============================================================
echo  TASK A: Multi-seed SCA-MobileNet on SCUT
echo  Seeds: 42, 0, 1, 7, 99
echo ============================================================

set SEEDS=42 0 1 7 99
set SEED_COUNT=0
set SEED_FAIL=0

for %%S in (%SEEDS%) do (
    set /a SEED_COUNT+=1
    echo.
    echo [A] Seed %%S (!SEED_COUNT!/5) - Started: %time%
    echo --------------------------------------------------------

    python train.py ^
        --model sca_mobilenet ^
        --dataset %SCUT_DATASET% ^
        --batch-size 16 ^
        --epochs 100 ^
        --lr 0.001 ^
        --feature-dim 1024 ^
        --eval-frequency 5 ^
        --database SCUT ^
        --loss-type adacos_only ^
        --sca-backbone mobilenetv3 ^
        --seed %%S ^
        --output-dir results/results_scut_seed%%S

    if errorlevel 1 (
        echo [FAILED] Seed %%S
        set /a SEED_FAIL+=1
    ) else (
        echo [OK] Seed %%S completed
    )
)

echo.
echo [A] Multi-seed complete: !SEED_COUNT! runs, !SEED_FAIL! failures
echo ============================================================

REM ============================================================
REM TASK F: Inference Benchmark (all 12 models)
REM Estimate: ~20-30 min
REM ============================================================
echo.
echo ============================================================
echo  TASK F: Inference Benchmark
echo ============================================================

python evaluation/inference_benchmark.py

if errorlevel 1 (
    echo [FAILED] Inference benchmark
) else (
    echo [OK] Inference benchmark completed
)

REM ============================================================
REM TASK G: Statistical Significance
REM Extracts EER from multi-seed results and runs analysis
REM Estimate: ~5 min
REM ============================================================
echo.
echo ============================================================
echo  TASK G: Extracting multi-seed EER and running statistics
echo ============================================================

python -c "
import json, os, sys
from pathlib import Path

seeds = [42, 0, 1, 7, 99]
base = Path('results')

eers = []
for s in seeds:
    d = base / f'results_scut_seed{s}_sca_mobilenet'
    f = d / 'training_metrics.json'
    if not f.exists():
        # Try alternative naming
        for alt in d.parent.glob(f'results_scut_seed{s}*'):
            f2 = alt / 'training_metrics.json'
            if f2.exists():
                f = f2
                break
    if f.exists():
        data = json.load(open(f))
        if 'epochs' in data and data['epochs']:
            best_eer = min(e['eer'] for e in data['epochs'] if 'eer' in e)
            eers.append((s, best_eer))
            print(f'  Seed {s}: EER = {best_eer:.4f}%%')
        else:
            print(f'  Seed {s}: no epoch data')
    else:
        print(f'  Seed {s}: results not found')

if len(eers) >= 2:
    import numpy as np
    vals = [e[1] for e in eers]
    mean_eer = np.mean(vals)
    std_eer = np.std(vals, ddof=1)
    print(f'\n  === SUMMARY ===')
    print(f'  SCA-MobileNet on SCUT: EER = {mean_eer:.2f}%% +/- {std_eer:.2f}%% ({len(eers)} seeds)')
    print(f'  Individual: {[f\"{e[0]}:{e[1]:.2f}\" for e in eers]}')

    # Save for paper
    out = {'model': 'SCA-MobileNet', 'dataset': 'SCUT', 'seeds': dict(eers),
           'mean_eer': float(mean_eer), 'std_eer': float(std_eer), 'n_seeds': len(eers)}
    outf = 'results/multi_seed_summary.json'
    json.dump(out, open(outf, 'w'), indent=2)
    print(f'  Saved to {outf}')
else:
    print(f'\n  Not enough seed results ({len(eers)}/5). Check training output above.')
"

if errorlevel 1 (
    echo [FAILED] Statistical analysis
) else (
    echo [OK] Statistical analysis completed
)

REM ============================================================
REM DONE
REM ============================================================
echo.
echo ============================================================
echo  ALL TASKS COMPLETE
echo  Finished: %date% %time%
echo ============================================================
echo.
echo  Results:
echo    A: results/results_scut_seed{42,0,1,7,99}_sca_mobilenet/
echo    F: benchmark_results/
echo    G: results/multi_seed_summary.json
echo.
echo  Next: Add results to paper (LaTeX)
echo ============================================================

pause
