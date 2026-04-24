@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

REM ============================================================
REM Statistical Significance: 5 seeds x 5 models = 25 runs
REM Models: SCA-3M, SCA-12M, RSNet, FGFNet, GSCL ResNet-18
REM Seeds:  42, 0, 1, 7, 99
REM Dataset: Internal (final_dataset_openset)
REM ============================================================

set "DATASET=datasets/final_dataset_openset"
set "OUT=results/results_5seeds"
set FAIL_COUNT=0
set DONE_COUNT=0
set TOTAL=25

echo ============================================================
echo  5-Seed Statistical Significance Experiment
echo  5 models x 5 seeds = 25 runs
echo  Started: %date% %time%
echo ============================================================

REM ==================== SCA-MobileNet 3M ====================
for %%S in (42 0 1 7 99) do (
    set /a DONE_COUNT+=1
    echo.
    echo [!DONE_COUNT!/%TOTAL%] SCA-3M seed=%%S - Started: %time%
    echo --------------------------------------------------------
    python train.py ^
        --model sca_mobilenet --sca-backbone mobilenetv3 ^
        --dataset "%DATASET%" ^
        --output-dir "%OUT%/sca_3m_s%%S" ^
        --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
        --batch-size 32 --epochs 100 --lr 0.001 ^
        --database default --eval-frequency 5 --seed %%S
    if errorlevel 1 (echo [FAILED] SCA-3M s%%S & set /a FAIL_COUNT+=1) else (echo [OK] SCA-3M s%%S)
)

REM ==================== SCA-MobileNet 12M ====================
for %%S in (42 0 1 7 99) do (
    set /a DONE_COUNT+=1
    echo.
    echo [!DONE_COUNT!/%TOTAL%] SCA-12M seed=%%S - Started: %time%
    echo --------------------------------------------------------
    python train.py ^
        --model sca_mobilenet --sca-backbone mobilenetv3 ^
        --no-bottleneck --ca-reduction 32 ^
        --dataset "%DATASET%" ^
        --output-dir "%OUT%/sca_12m_s%%S" ^
        --loss-type adacos_only --feature-dim 1024 --dropout 0.3 ^
        --batch-size 32 --epochs 100 --lr 0.001 ^
        --database default --eval-frequency 5 --seed %%S
    if errorlevel 1 (echo [FAILED] SCA-12M s%%S & set /a FAIL_COUNT+=1) else (echo [OK] SCA-12M s%%S)
)

REM ==================== RSNet ====================
for %%S in (42 0 1 7 99) do (
    set /a DONE_COUNT+=1
    echo.
    echo [!DONE_COUNT!/%TOTAL%] RSNet seed=%%S - Started: %time%
    echo --------------------------------------------------------
    python train.py ^
        --model rsnet ^
        --dataset "%DATASET%" ^
        --output-dir "%OUT%/rsnet_s%%S" ^
        --feature-dim 1024 ^
        --batch-size 16 --epochs 100 --lr 0.001 ^
        --database default --eval-frequency 5 --seed %%S
    if errorlevel 1 (echo [FAILED] RSNet s%%S & set /a FAIL_COUNT+=1) else (echo [OK] RSNet s%%S)
)

REM ==================== FGFNet ====================
REM Note: train.py overrides lr 0.001 -> 1e-4 for FGFNet
for %%S in (42 0 1 7 99) do (
    set /a DONE_COUNT+=1
    echo.
    echo [!DONE_COUNT!/%TOTAL%] FGFNet seed=%%S - Started: %time%
    echo --------------------------------------------------------
    python train.py ^
        --model fgfnet ^
        --dataset "%DATASET%" ^
        --output-dir "%OUT%/fgfnet_s%%S" ^
        --feature-dim 640 ^
        --batch-size 16 --epochs 100 --lr 0.001 ^
        --database default --eval-frequency 5 --seed %%S
    if errorlevel 1 (echo [FAILED] FGFNet s%%S & set /a FAIL_COUNT+=1) else (echo [OK] FGFNet s%%S)
)

REM ==================== GSCL ResNet-18 ====================
for %%S in (42 0 1 7 99) do (
    set /a DONE_COUNT+=1
    echo.
    echo [!DONE_COUNT!/%TOTAL%] GSCL seed=%%S - Started: %time%
    echo --------------------------------------------------------
    python train.py ^
        --model gscl --gscl-backbone resnet18 ^
        --dataset "%DATASET%" ^
        --output-dir "%OUT%/gscl_s%%S" ^
        --feature-dim 512 ^
        --batch-size 64 --epochs 100 --lr 0.01 ^
        --database default --eval-frequency 5 --seed %%S
    if errorlevel 1 (echo [FAILED] GSCL s%%S & set /a FAIL_COUNT+=1) else (echo [OK] GSCL s%%S)
)

echo.
echo ============================================================
echo  ALL COMPLETE: %date% %time%
echo  Failed: !FAIL_COUNT!/%TOTAL%
echo ============================================================

REM ==================== Auto-analyze results ====================
echo.
echo Running statistical analysis...
python -c "
import json, numpy as np
from pathlib import Path

models = {
    'SCA-3M':  ('sca_3m',  [42,0,1,7,99]),
    'SCA-12M': ('sca_12m', [42,0,1,7,99]),
    'RSNet':   ('rsnet',   [42,0,1,7,99]),
    'FGFNet':  ('fgfnet',  [42,0,1,7,99]),
    'GSCL':    ('gscl',    [42,0,1,7,99]),
}

print()
print(f'{'Model':<12} | {'Mean EER':>10} | {'Std':>8} | {'Min':>8} | {'Max':>8} | Seeds')
print('-' * 70)
for name, (prefix, seeds) in models.items():
    eers = []
    for s in seeds:
        p = Path(f'results/results_5seeds/{prefix}_s{s}_sca_mobilenet/training_metrics.json')
        if not p.exists():
            for suffix in ['_rsnet', '_fgfnet', '_gscl']:
                p2 = Path(f'results/results_5seeds/{prefix}_s{s}{suffix}/training_metrics.json')
                if p2.exists():
                    p = p2
                    break
        if p.exists():
            d = json.load(open(p))
            best = min(d['epochs'], key=lambda e: e.get('eer',1))
            eers.append(best['eer']*100)
    if eers:
        arr = np.array(eers)
        print(f'{name:<12} | {arr.mean():>9.4f}%% | {arr.std():>7.4f} | {arr.min():>7.4f} | {arr.max():>7.4f} | {len(eers)}/5')
    else:
        print(f'{name:<12} | NO RESULTS')
print()
"

pause
