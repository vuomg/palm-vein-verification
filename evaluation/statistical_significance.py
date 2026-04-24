"""
Statistical Significance Analysis — 5 Random Seeds
Report EER mean ± std và Wilcoxon test
Dùng cho paper Q1 — chứng minh cải thiện có ý nghĩa thống kê
"""

import numpy as np
import json
import os
from pathlib import Path
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ============================================================
# CẤU HÌNH
# ============================================================
# Sau khi chạy xong 5 seed, điền kết quả EER vào đây
# Format: {"model_name": [eer_seed42, eer_seed0, eer_seed1, eer_seed7, eer_seed99]}
# Đơn vị: % (ví dụ 0.89, không phải 0.0089)

SEEDS = [42, 0, 1, 7, 99]

# TODO: điền kết quả thực tế sau khi train xong
EER_RESULTS = {
    "RSNet"              : [None, None, None, None, None],
    "ResNet-50"          : [3.78, None, None, None, None],
    "EfficientNet-B0"    : [1.25, None, None, None, None],
    "MobileNetV3-Small"  : [1.14, None, None, None, None],
    "DeiT-Tiny"          : [None, None, None, None, None],
    "MobileViT-S"        : [None, None, None, None, None],
    "Swin-Tiny"          : [None, None, None, None, None],
    "MPSNet"             : [0.99, None, None, None, None],
    "FGFNet"             : [None, None, None, None, None],
    "GSCL (ResNet-18)"   : [None, None, None, None, None],
    "EUSIPCO-DenseNet161": [None, None, None, None, None],
    "SCA-MobileNet"      : [0.89, None, None, None, None],
}

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results_sca_v2_sca_mobilenet"

OUTPUT_DIR = str(RESULTS_ROOT / "statistical_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PHÂN TÍCH THỐNG KÊ
# ============================================================
def compute_statistics(eer_results):
    """Tính mean, std, CI95 cho mỗi model"""
    stats_dict = {}
    for model, eers in eer_results.items():
        valid = [e for e in eers if e is not None]
        if len(valid) == 0:
            stats_dict[model] = {
                "n": 0, "mean": 0.0, "std": 0.0, "ci95": 0.0,
                "min": 0.0, "max": 0.0, "values": [],
            }
            continue
        arr = np.array(valid)
        n = len(arr)
        mean = np.mean(arr)
        std  = np.std(arr, ddof=1) if n > 1 else 0.0
        ci95 = stats.t.ppf(0.975, df=max(n-1, 1)) * std / np.sqrt(n) if n > 1 else 0.0
        stats_dict[model] = {
            "n"       : n,
            "mean"    : mean,
            "std"     : std,
            "ci95"    : ci95,
            "min"     : np.min(arr),
            "max"     : np.max(arr),
            "values"  : valid,
        }
    return stats_dict


def wilcoxon_vs_ours(eer_results, ours_key="SCA-MobileNet"):
    """
    Wilcoxon signed-rank test: SCA-MobileNet vs mỗi baseline
    H0: không có sự khác biệt có ý nghĩa thống kê
    """
    ours = [e for e in eer_results[ours_key] if e is not None]
    results = {}

    for model, eers in eer_results.items():
        if model == ours_key:
            continue
        valid = [e for e in eers if e is not None]
        if len(valid) < 2 or len(ours) < 2:
            results[model] = {"p_value": None, "significant": None, "note": "Insufficient data"}
            continue

        # Cần cùng số samples
        n = min(len(valid), len(ours))
        try:
            stat, p = stats.wilcoxon(ours[:n], valid[:n], alternative="less")
            results[model] = {
                "statistic"   : float(stat),
                "p_value"     : float(p),
                "significant" : p < 0.05,
                "note"        : "p<0.05: SCA-MobileNet significantly better" if p < 0.05
                                else "p≥0.05: not significant"
            }
        except Exception as ex:
            results[model] = {"p_value": None, "significant": None, "note": str(ex)}

    return results


# ============================================================
# FIGURE: EER mean ± std bar chart
# ============================================================
def plot_eer_with_errorbar(stats_dict, output_dir):
    models  = list(stats_dict.keys())
    means   = [stats_dict[m]["mean"] for m in models]
    stds    = [stats_dict[m]["std"]  for m in models]
    cis     = [stats_dict[m]["ci95"] for m in models]
    has_data = [stats_dict[m]["n"] > 0 for m in models]

    colors = []
    for m, hd in zip(models, has_data):
        if not hd:
            colors.append("#cccccc")
        elif m == "SCA-MobileNet":
            colors.append("#e63946")
        else:
            colors.append("#457b9d")

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(models))

    bars = ax.bar(x, means, yerr=cis, capsize=5, color=colors,
                  alpha=0.85, edgecolor="white", linewidth=1.2,
                  error_kw=dict(ecolor="black", lw=1.5))

    for i, bar in enumerate(bars):
        if not has_data[i]:
            bar.set_hatch("//")
            bar.set_edgecolor("#999999")

    for i, (m, s, ci, hd) in enumerate(zip(means, stds, cis, has_data)):
        if hd:
            ax.text(i, m + ci + 0.05, f"{m:.2f}\u00b1{s:.2f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        else:
            ax.text(i, 0.15, "N/A",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color="#999999", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("EER (%)", fontsize=12)
    ax.set_title("EER Comparison (Mean \u00b1 95% CI over 5 Random Seeds)", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    if "SCA-MobileNet" in models and stats_dict["SCA-MobileNet"]["n"] > 0:
        ax.axhline(y=stats_dict["SCA-MobileNet"]["mean"],
                   color="#e63946", linestyle="--", alpha=0.4, linewidth=1)

    plt.tight_layout()
    out = os.path.join(output_dir, "eer_mean_std_comparison.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# EXPORT: LaTeX table
# ============================================================
def export_latex_table(stats_dict, wilcoxon_results, output_dir):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Statistical Comparison of EER (\%) over 5 Random Seeds}")
    lines.append(r"\label{tab:statistical}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r"Model & Mean & Std & 95\% CI & $p$-value vs Ours \\")
    lines.append(r"\hline")

    for model, s in stats_dict.items():
        p_str = "—"
        if model != "SCA-MobileNet" and model in wilcoxon_results:
            pv = wilcoxon_results[model]["p_value"]
            if pv is not None:
                p_str = f"{pv:.4f}{'*' if pv < 0.05 else ''}"

        bold = r"\textbf" if model == "SCA-MobileNet" else ""
        if s["n"] == 0:
            lines.append(f"{model} & -- & -- & -- & {p_str} \\\\")
        elif bold:
            lines.append(
                f"{bold}{{{model}}} & {bold}{{{s['mean']:.2f}}} & "
                f"{bold}{{{s['std']:.2f}}} & {bold}{{{s['ci95']:.2f}}} & {p_str} \\\\"
            )
        else:
            lines.append(
                f"{model} & {s['mean']:.2f} & {s['std']:.2f} & {s['ci95']:.2f} & {p_str} \\\\"
            )

    lines.append(r"\hline")
    lines.append(r"\multicolumn{5}{l}{\small * $p < 0.05$: SCA-MobileNet significantly better} \\")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = os.path.join(output_dir, "statistical_table.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: {out}")


# ============================================================
# SCRIPT TRAIN NHIỀU SEED (chạy từ command line)
# ============================================================
TRAIN_SCRIPT_TEMPLATE = '''
#!/bin/bash
# Chạy script này để train SCA-MobileNet với 5 seed
# Điều chỉnh đường dẫn train.py cho đúng

SEEDS=(42 0 1 7 99)
for SEED in "${SEEDS[@]}"; do
    echo "Training with seed=$SEED..."
    python train.py \\
        --seed $SEED \\
        --output_dir results/seed_$SEED \\
        --epochs 100
    echo "Done seed=$SEED"
done
echo "All seeds completed!"
'''

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Save bash script
    bash_out = os.path.join(OUTPUT_DIR, "run_5_seeds.sh")
    with open(bash_out, "w", encoding="utf-8") as f:
        f.write(TRAIN_SCRIPT_TEMPLATE)
    print(f"Saved bash script: {bash_out}")

    # Phân tích với dữ liệu hiện có (seed 42)
    print("\nPhân tích với dữ liệu seed=42 (sẽ cập nhật sau khi có đủ 5 seeds):")
    stats_dict = compute_statistics(EER_RESULTS)

    if stats_dict:
        print("\n--- Thống kê ---")
        for model, s in stats_dict.items():
            if s["n"] == 0:
                print(f"  {model:<25}: N/A (no data yet)")
            else:
                print(f"  {model:<25}: EER = {s['mean']:.2f} ± {s['std']:.2f}% (n={s['n']})")

        wilcoxon_results = wilcoxon_vs_ours(EER_RESULTS)
        print("\n--- Wilcoxon test vs SCA-MobileNet ---")
        for model, r in wilcoxon_results.items():
            print(f"  vs {model:<22}: {r['note']}")

        plot_eer_with_errorbar(stats_dict, OUTPUT_DIR)
        export_latex_table(stats_dict, wilcoxon_results, OUTPUT_DIR)

    print("\n✅ Sau khi train đủ 5 seed, điền EER_RESULTS và chạy lại script này.")
    print("   Kết quả sẽ tự động tạo figure + LaTeX table cho paper.")
