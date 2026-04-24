"""
Rejection Analysis Script for SCA-MobileNet
Phân tích 20 file rejection JSON qua các epoch
Dùng cho paper Q1 — tạo figures chuẩn publication
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# ============================================================
# CẤU HÌNH — chỉnh đường dẫn này
# ============================================================
REJECTION_DIR = r"results_sca_v2_sca_mobilenet\rejection_analysis"
OUTPUT_DIR    = r"results_sca_v2_sca_mobilenet\rejection_analysis\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FAR_KEYS   = ["far_001", "far_01", "far_1"]
FAR_LABELS = ["FAR=0.01%", "FAR=0.1%", "FAR=1%"]
COLORS     = ["#e63946", "#f4a261", "#2a9d8f"]

# ============================================================
# LOAD DATA
# ============================================================
def load_all_epochs(folder):
    files = sorted(glob.glob(os.path.join(folder, "epoch_*_rejections.json")),
                   key=lambda x: int(x.split("epoch_")[1].split("_")[0]))
    data = []
    for f in files:
        with open(f) as fp:
            data.append(json.load(fp))
    print(f"Loaded {len(data)} epoch files.")
    return data

all_data = load_all_epochs(REJECTION_DIR)
if len(all_data) == 0:
    print(f"⚠️ No rejection JSON files found in: {REJECTION_DIR}")
    print("   Expected pattern: epoch_*_rejections.json")
    print("   Run evaluation/rejection export first, then re-run this script.")
    raise SystemExit(0)

epochs = [d["epoch"] for d in all_data]

# ============================================================
# FIGURE 1: Rejection Rate qua các Epoch (3 FAR levels)
# ============================================================
def plot_rejection_rate_over_epochs(all_data, epochs):
    fig, ax = plt.subplots(figsize=(10, 5))
    for far_key, label, color in zip(FAR_KEYS, FAR_LABELS, COLORS):
        rates = [d["analysis"][far_key]["rejection_rate"] * 100 for d in all_data]
        ax.plot(epochs, rates, label=label, color=color, linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Rejection Rate (%)", fontsize=13)
    ax.set_title("System-level Rejection Rate over Training Epochs", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig1_rejection_rate_over_epochs.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

plot_rejection_rate_over_epochs(all_data, epochs)

# ============================================================
# FIGURE 2: Số user bị reject qua các Epoch
# ============================================================
def plot_n_rejected_users(all_data, epochs):
    fig, ax = plt.subplots(figsize=(10, 5))
    for far_key, label, color in zip(FAR_KEYS, FAR_LABELS, COLORS):
        counts = [d["analysis"][far_key]["n_rejected_users"] for d in all_data]
        ax.plot(epochs, counts, label=label, color=color, linewidth=2, marker="s", markersize=4)

    total_users = all_data[0]["analysis"]["far_001"]["n_total_users"]
    ax.axhline(y=total_users * 0.05, color="gray", linestyle="--", alpha=0.5, label="5% threshold")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Number of Rejected Users", fontsize=13)
    ax.set_title(f"Number of Rejected Users over Training (Total: {total_users})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig2_n_rejected_users.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

plot_n_rejected_users(all_data, epochs)

# ============================================================
# FIGURE 3: Persistent Failure Cases — user bị reject nhiều nhất
# ============================================================
def find_persistent_failures(all_data, far_key="far_001", top_n=15):
    """Đếm số epoch mỗi user bị reject"""
    reject_count = defaultdict(int)
    reject_rate_sum = defaultdict(float)

    for d in all_data:
        for user in d["analysis"][far_key]["rejected_users"]:
            lbl = user["label"]
            reject_count[lbl] += 1
            reject_rate_sum[lbl] += user["rejection_rate"]

    # Sort by số epoch bị reject
    sorted_users = sorted(reject_count.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return sorted_users, reject_rate_sum

def plot_persistent_failures(all_data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Persistent Failure Cases — Users Rejected Most Frequently", fontsize=14, fontweight="bold")

    for ax, far_key, label, color in zip(axes, FAR_KEYS, FAR_LABELS, COLORS):
        top_users, rate_sum = find_persistent_failures(all_data, far_key=far_key, top_n=15)
        labels_u = [f"ID-{u[0]}" for u in top_users]
        counts = [u[1] for u in top_users]

        bars = ax.barh(labels_u[::-1], counts[::-1], color=color, alpha=0.85, edgecolor="white")
        ax.set_xlabel("# Epochs Rejected (out of 20)", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlim(0, 20)
        ax.axvline(x=20, color="gray", linestyle="--", alpha=0.4)
        ax.grid(True, axis="x", alpha=0.3)

        for bar, cnt in zip(bars, counts[::-1]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{cnt}/20", va="center", fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig3_persistent_failure_cases.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

plot_persistent_failures(all_data)

# ============================================================
# FIGURE 4: Score Distribution của Failure Cases (epoch 100)
# ============================================================
def plot_failure_score_distribution(all_data, far_key="far_001"):
    last_epoch = all_data[-1]  # epoch 100
    users = last_epoch["analysis"][far_key]["rejected_users"]

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(users))
    width = 0.35

    avg_scores = [u["avg_genuine_score"] for u in users]
    min_scores = [u["min_genuine_score"] for u in users]
    labels_u   = [f"ID-{u['label']}" for u in users]
    threshold  = last_epoch["analysis"][far_key]["threshold"]

    bars1 = ax.bar(x - width/2, avg_scores, width, label="Avg Genuine Score",
                   color="#2a9d8f", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, min_scores, width, label="Min Genuine Score",
                   color="#e63946", alpha=0.85, edgecolor="white")

    ax.axhline(y=threshold, color="navy", linestyle="--", linewidth=2,
               label=f"Threshold ({threshold:.3f})")

    ax.set_xlabel("User Identity", fontsize=12)
    ax.set_ylabel("Cosine Similarity Score", fontsize=12)
    ax.set_title(f"Genuine Score Distribution of Rejected Users at Epoch 100 ({FAR_LABELS[FAR_KEYS.index(far_key)]})",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_u, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, f"fig4_score_distribution_{far_key}.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

plot_failure_score_distribution(all_data, far_key="far_001")

# ============================================================
# FIGURE 5: Avg Genuine Score của Persistent Failures qua Epoch
# ============================================================
def plot_score_evolution_persistent(all_data, epochs, far_key="far_001", top_n=5):
    """Track avg genuine score của top persistent failures qua 20 epoch"""
    top_users, _ = find_persistent_failures(all_data, far_key=far_key, top_n=top_n)
    target_labels = [u[0] for u in top_users]

    # Build score trajectory
    trajectories = {lbl: [] for lbl in target_labels}
    epoch_list = []

    for d in all_data:
        ep = d["epoch"]
        epoch_list.append(ep)
        users_this_epoch = {u["label"]: u for u in d["analysis"][far_key]["rejected_users"]}
        for lbl in target_labels:
            if lbl in users_this_epoch:
                trajectories[lbl].append(users_this_epoch[lbl]["avg_genuine_score"])
            else:
                trajectories[lbl].append(None)

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.cm.Set2(np.linspace(0, 1, top_n))

    for (lbl, scores), color in zip(trajectories.items(), cmap):
        valid_epochs = [e for e, s in zip(epoch_list, scores) if s is not None]
        valid_scores = [s for s in scores if s is not None]
        ax.plot(valid_epochs, valid_scores, label=f"ID-{lbl}", color=color,
                linewidth=2, marker="o", markersize=5)

    threshold_vals = [d["analysis"][far_key]["threshold"] for d in all_data]
    ax.plot(epoch_list, threshold_vals, "k--", linewidth=1.5, label="Decision Threshold", alpha=0.7)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Avg Genuine Score", fontsize=12)
    ax.set_title(f"Score Evolution of Top-{top_n} Persistent Failure Users ({FAR_LABELS[FAR_KEYS.index(far_key)]})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epoch_list)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, f"fig5_score_evolution_{far_key}.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

plot_score_evolution_persistent(all_data, epochs, far_key="far_001", top_n=5)

# ============================================================
# EXPORT: Summary Table (CSV)
# ============================================================
def export_summary_csv(all_data, epochs):
    import csv
    out = os.path.join(OUTPUT_DIR, "rejection_summary.csv")
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Threshold_001", "N_Rejected_001", "Rate_001(%)",
                                  "Threshold_01",  "N_Rejected_01",  "Rate_01(%)",
                                  "Threshold_1",   "N_Rejected_1",   "Rate_1(%)"])
        for d in all_data:
            ep = d["epoch"]
            row = [ep]
            for fk in FAR_KEYS:
                a = d["analysis"][fk]
                row += [f"{a['threshold']:.4f}", a["n_rejected_users"],
                        f"{a['rejection_rate']*100:.2f}"]
            writer.writerow(row)
    print(f"Saved: {out}")

export_summary_csv(all_data, epochs)

print("\n✅ Hoàn tất! Tất cả figures đã được lưu vào:", OUTPUT_DIR)
print("Files tạo ra:")
print("  fig1_rejection_rate_over_epochs.png  — Training curve rejection rate")
print("  fig2_n_rejected_users.png            — Số user bị reject qua epoch")
print("  fig3_persistent_failure_cases.png    — Top users bị reject nhiều nhất")
print("  fig4_score_distribution_far_001.png  — Score distribution epoch 100")
print("  fig5_score_evolution_far_001.png     — Score trajectory qua epoch")
print("  rejection_summary.csv               — Bảng tổng hợp cho paper")
