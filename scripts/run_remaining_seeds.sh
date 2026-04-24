#!/bin/bash
# Run remaining seeds (0, 1, 7, 99) + benchmark + statistics
# Seed 42 already completed

cd "$(dirname "$0")/.."

DATASET="datasets/SCUT_dataset_openset"
COMMON_ARGS="--model sca_mobilenet --dataset $DATASET --batch-size 16 --epochs 100 --lr 0.001 --feature-dim 1024 --eval-frequency 5 --database SCUT --loss-type adacos_only --sca-backbone mobilenetv3"

for SEED in 0 1 7 99; do
    echo "=========================================="
    echo " Seed $SEED - Started: $(date)"
    echo "=========================================="
    python train.py $COMMON_ARGS --seed $SEED --output-dir results/results_scut_seed${SEED}
    echo "[DONE] Seed $SEED - $(date)"
done

echo "=========================================="
echo " Inference Benchmark"
echo "=========================================="
python evaluation/inference_benchmark.py
echo "[DONE] Benchmark - $(date)"

echo "=========================================="
echo " Statistical Analysis"
echo "=========================================="
python -c "
import json, numpy as np
from pathlib import Path

seeds = [42, 0, 1, 7, 99]
base = Path('results')
eers = []

for s in seeds:
    for d in base.glob(f'results_scut_seed{s}*'):
        f = d / 'training_metrics.json'
        if f.exists():
            data = json.load(open(f))
            if 'epochs' in data and data['epochs']:
                best_eer = min(e['eer'] for e in data['epochs'] if 'eer' in e)
                eers.append((s, best_eer * 100))
                print(f'  Seed {s}: EER = {best_eer*100:.4f}%')
            break

if len(eers) >= 2:
    vals = [e[1] for e in eers]
    mean_eer = np.mean(vals)
    std_eer = np.std(vals, ddof=1)
    print(f'\n  === SUMMARY ===')
    print(f'  SCA-MobileNet on SCUT: EER = {mean_eer:.2f}% +/- {std_eer:.2f}% ({len(eers)} seeds)')

    out = {'model': 'SCA-MobileNet', 'dataset': 'SCUT', 'seeds': dict(eers),
           'mean_eer': float(mean_eer), 'std_eer': float(std_eer), 'n_seeds': len(eers)}
    json.dump(out, open('results/multi_seed_summary.json', 'w'), indent=2)
    print(f'  Saved to results/multi_seed_summary.json')
else:
    print(f'  Not enough seed results ({len(eers)}/5)')
"

echo "=========================================="
echo " ALL COMPLETE - $(date)"
echo "=========================================="
