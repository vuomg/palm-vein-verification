"""
Bubble Chart: Model Efficiency vs Accuracy
Trục X: FLOPs (G) — độ phức tạp tính toán
Trục Y: Accuracy (100% - EER) — độ chính xác
Kích thước bong bóng: Parameters (M)
Output: paper/fig_efficiency_bubble.png
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import os
from pathlib import Path

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "paper"

MODELS = {
    "SCA-MobileNet\n(Ours)": {
        "params": 3.19, "flops": 0.13,
        "eer_internal": 0.89, "eer_tongji": 0.06, "eer_scut": 1.48, "eer_vera": 2.76,
        "is_ours": True,
    },
    "MobileNetV3\n(Base)": {
        "params": 1.52, "flops": 0.06,
        "eer_internal": None, "eer_tongji": 0.29, "eer_scut": 2.33, "eer_vera": 3.62,
        "is_ours": False,
    },
    "MPSNet": {
        "params": 2.99, "flops": 0.14,
        "eer_internal": None, "eer_tongji": 0.26, "eer_scut": 1.57, "eer_vera": 4.87,
        "is_ours": False,
    },
    "EfficientNet\n-B0": {
        "params": 4.01, "flops": 0.38,
        "eer_internal": None, "eer_tongji": 0.57, "eer_scut": 3.02, "eer_vera": 7.21,
        "is_ours": False,
    },
    "DeiT-Tiny": {
        "params": 5.52, "flops": 1.07,
        "eer_internal": None, "eer_tongji": 1.93, "eer_scut": 5.04, "eer_vera": 6.99,
        "is_ours": False,
    },
    "FGFNet": {
        "params": 5.62, "flops": 6.37,
        "eer_internal": 3.93, "eer_tongji": 5.52, "eer_scut": 5.53, "eer_vera": 9.68,
        "is_ours": False,
    },
    "MobileViT-S": {
        "params": 4.94, "flops": 1.83,
        "eer_internal": None, "eer_tongji": 1.32, "eer_scut": 4.94, "eer_vera": 8.79,
        "is_ours": False,
    },
    "RSNet": {
        "params": 6.23, "flops": 1.17,
        "eer_internal": 2.82, "eer_tongji": 1.40, "eer_scut": 6.37, "eer_vera": 8.50,
        "is_ours": False,
    },
    "GSCL": {
        "params": 11.23, "flops": 1.82,
        "eer_internal": 2.16, "eer_tongji": 1.73, "eer_scut": 3.42, "eer_vera": 7.17,
        "is_ours": False,
    },
    "ResNet50": {
        "params": 23.51, "flops": 4.13,
        "eer_internal": None, "eer_tongji": 0.72, "eer_scut": 3.38, "eer_vera": 8.89,
        "is_ours": False,
    },
    "DenseNet161": {
        "params": 28.74, "flops": 7.95,
        "eer_internal": None, "eer_tongji": 0.39, "eer_scut": 1.51, "eer_vera": 7.14,
        "is_ours": False,
    },
    "Swin-Tiny": {
        "params": 27.52, "flops": 4.37,
        "eer_internal": None, "eer_tongji": 1.13, "eer_scut": 3.71, "eer_vera": 5.69,
        "is_ours": False,
    },
}

# Try to load benchmark results for latency data
BENCHMARK_FILE = PROJECT_ROOT / "benchmark_results" / "benchmark_results.json"


def load_benchmark():
    if BENCHMARK_FILE.exists():
        return json.load(open(BENCHMARK_FILE))
    return None


def plot_bubble_chart(dataset_key, dataset_label, ax):
    names, flops, accuracy, params, colors, edges = [], [], [], [], [], []

    for name, data in MODELS.items():
        eer = data.get(dataset_key)
        if eer is None:
            continue
        names.append(name)
        flops.append(data["flops"])
        accuracy.append(100.0 - eer)
        params.append(data["params"])
        if data["is_ours"]:
            colors.append("#e74c3c")
            edges.append("#c0392b")
        else:
            colors.append("#3498db")
            edges.append("#2980b9")

    sizes = np.array(params)
    sizes_normalized = (sizes / sizes.max()) * 800 + 80

    ax.scatter(flops, accuracy, s=sizes_normalized, c=colors,
               edgecolors=edges, linewidths=1.5, alpha=0.75, zorder=5)

    for i, name in enumerate(names):
        offset_y = 0.15
        if "Ours" in name:
            ax.annotate(name, (flops[i], accuracy[i]),
                        textcoords="offset points", xytext=(0, 18),
                        ha='center', fontsize=8, fontweight='bold',
                        color='#c0392b',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffeaea',
                                  edgecolor='#e74c3c', alpha=0.9))
        else:
            ax.annotate(name, (flops[i], accuracy[i]),
                        textcoords="offset points", xytext=(0, 14),
                        ha='center', fontsize=7, color='#2c3e50', alpha=0.85)

    ax.set_xlabel("FLOPs (G)", fontsize=11)
    ax.set_ylabel("Accuracy (100% - EER)", fontsize=11)
    ax.set_title(dataset_label, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.3, max(flops) * 1.15)


def plot_bubble_latency(ax):
    """If benchmark data exists, plot latency-based bubble chart."""
    bench = load_benchmark()
    if bench is None:
        ax.text(0.5, 0.5, "Benchmark data not available\nRun: python evaluation/inference_benchmark.py",
                ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
        ax.set_title("Latency vs Accuracy (pending)", fontsize=12)
        return

    names, latencies, accuracy, params, colors, edges = [], [], [], [], [], []
    for name, data in MODELS.items():
        eer = data.get("eer_scut")
        if eer is None:
            continue
        clean_name = name.replace("\n", " ").replace("(Ours)", "").strip()
        bench_key = None
        for k in bench:
            if clean_name.lower().replace("-", "").replace(" ", "") in k.lower().replace("-", "").replace(" ", ""):
                bench_key = k
                break
        if bench_key is None:
            continue

        gpu_data = bench[bench_key].get("gpu", bench[bench_key].get("cpu", {}))
        lat = gpu_data.get("mean_ms", None)
        if lat is None:
            continue

        names.append(name)
        latencies.append(lat)
        accuracy.append(100.0 - eer)
        params.append(data["params"])
        colors.append("#e74c3c" if data["is_ours"] else "#3498db")
        edges.append("#c0392b" if data["is_ours"] else "#2980b9")

    if not names:
        ax.text(0.5, 0.5, "No matching benchmark entries",
                ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
        return

    sizes = np.array(params)
    sizes_normalized = (sizes / sizes.max()) * 800 + 80

    ax.scatter(latencies, accuracy, s=sizes_normalized, c=colors,
               edgecolors=edges, linewidths=1.5, alpha=0.75, zorder=5)

    for i, name in enumerate(names):
        if "Ours" in name:
            ax.annotate(name, (latencies[i], accuracy[i]),
                        textcoords="offset points", xytext=(0, 18),
                        ha='center', fontsize=8, fontweight='bold', color='#c0392b',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffeaea',
                                  edgecolor='#e74c3c', alpha=0.9))
        else:
            ax.annotate(name, (latencies[i], accuracy[i]),
                        textcoords="offset points", xytext=(0, 14),
                        ha='center', fontsize=7, color='#2c3e50', alpha=0.85)

    ax.set_xlabel("Latency (ms/image, GPU)", fontsize=11)
    ax.set_ylabel("Accuracy (100% - EER)", fontsize=11)
    ax.set_title("SCUT — Latency vs Accuracy", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.invert_xaxis()


def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # Bubble charts for each dataset (FLOPs-based)
    datasets = [
        ("eer_tongji", "TONGJI — FLOPs vs Accuracy"),
        ("eer_scut", "SCUT — FLOPs vs Accuracy"),
        ("eer_vera", "VERA — FLOPs vs Accuracy"),
    ]

    for ax, (key, label) in zip([axes[0, 0], axes[0, 1], axes[1, 0]], datasets):
        plot_bubble_chart(key, label, ax)

    # Latency-based chart (bottom-right)
    plot_bubble_latency(axes[1, 1])

    # Legend for bubble sizes
    legend_sizes = [1, 5, 15, 30]
    max_param = max(d["params"] for d in MODELS.values())
    legend_bubbles = []
    for s in legend_sizes:
        legend_bubbles.append(
            axes[0, 0].scatter([], [], s=(s / max_param) * 800 + 80,
                               c='white', edgecolors='gray', linewidths=1,
                               label=f'{s}M params')
        )
    axes[0, 0].legend(handles=legend_bubbles, scatterpoints=1, frameon=True,
                       title="Model Size", loc='lower left', fontsize=8,
                       title_fontsize=9)

    fig.suptitle(
        "Model Efficiency Analysis: Accuracy vs Computational Cost\n"
        "Red = SCA-MobileNet (Ours) | Blue = Baselines | Bubble size = Parameters",
        fontsize=13, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = OUTPUT_DIR / "fig_efficiency_bubble.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_path}")

    out_path2 = OUTPUT_DIR / "fig_efficiency_bubble.pdf"
    plt.savefig(out_path2, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_path2}")

    plt.close()


if __name__ == "__main__":
    main()
