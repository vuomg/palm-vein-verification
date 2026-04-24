"""
Regenerate fig_roc.png and fig_det.png from the 3M SCA-MobileNet checkpoint.
Uses the internal test dataset with BiometricEvaluator.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'models'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from scipy.stats import norm

from SCA_MobileNet.model import SCAMobileNet
from biometric.metrics import BiometricEvaluator

CHECKPOINT = PROJECT_ROOT / 'results' / 'results_sca_sca_mobilenet' / 'best_sca_mobilenet_model_eer.pth'
DATASET_DIR = PROJECT_ROOT / 'datasets' / 'final_dataset_openset' / 'test'
PAPER_DIR = PROJECT_ROOT / 'paper'


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        idx = 0
        for d in sorted(self.data_dir.iterdir()):
            if not d.is_dir():
                continue
            self.class_to_idx[d.name] = idx
            for f in sorted(d.glob('*')):
                if f.suffix.lower() in ('.png', '.jpg', '.bmp'):
                    self.samples.append((str(f), idx))
            idx += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, label


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model (3M: bottleneck=True, ca_reduction=8)
    model = SCAMobileNet(
        embedding_size=1024,
        class_size=None,
        pretrained=False,
        only_embeddings=True,
        use_stn=True,
        use_ca=True,
        use_spp=True,
        dropout=0.3,
        use_bottleneck=True,
        ca_reduction=8
    ).to(device)

    ckpt = torch.load(str(CHECKPOINT), map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    print("Model loaded from", CHECKPOINT.name)

    # Dataset
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = SimpleDataset(DATASET_DIR, test_transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"Test set: {len(dataset)} images, {len(dataset.class_to_idx)} classes")

    # Extract embeddings
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Extracting embeddings'):
            images = images.to(device)
            out = model(images)
            if isinstance(out, tuple):
                out = out[0]
            out = torch.nn.functional.normalize(out, p=2, dim=1)
            all_emb.append(out.cpu().numpy())
            all_lbl.append(labels.numpy())
    embeddings = np.concatenate(all_emb)
    labels = np.concatenate(all_lbl)
    print(f"Embeddings shape: {embeddings.shape}")

    # Evaluate
    evaluator = BiometricEvaluator()
    results = evaluator.evaluate_verification(embeddings, labels)
    eer = results['EER'] * 100
    auc_val = results['AUC']
    tar_001 = results['TAR@0.01%FAR'] * 100
    tar_01 = results['TAR@0.1%FAR'] * 100
    tar_1 = results['TAR@1%FAR'] * 100
    d_prime = results['d_prime']

    print(f"\nEER: {eer:.4f}%")
    print(f"TAR@0.01%FAR: {tar_001:.2f}%")
    print(f"TAR@0.1%FAR: {tar_01:.2f}%")
    print(f"TAR@1%FAR: {tar_1:.2f}%")
    print(f"AUC: {auc_val:.6f}")
    print(f"D-prime: {d_prime:.2f}")

    genuine = evaluator.genuine_scores
    impostor = evaluator.imposter_scores

    # ========== ROC Curve ==========
    fig, ax = plt.subplots(figsize=(8, 7))

    y_true = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    y_scores = np.concatenate([genuine, impostor])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr * 100, tpr * 100, 'r-', linewidth=2.5, label=f'SCA-MobileNet (AUC={roc_auc:.4f})')
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Random chance')

    # Operating points
    for far_target, color, marker, name in [
        (0.0001, '#1565C0', 'o', f'FAR=0.01%: TAR={tar_001:.1f}%'),
        (0.001, '#7B1FA2', 's', f'FAR=0.1%: TAR={tar_01:.1f}%'),
        (0.01, '#E65100', 'D', f'FAR=1%: TAR={tar_1:.1f}%'),
    ]:
        idx = np.argmin(np.abs(fpr - far_target))
        ax.plot(fpr[idx]*100, tpr[idx]*100, marker=marker, color=color, markersize=10, zorder=5, label=name)
        ax.annotate(f'{tpr[idx]*100:.1f}%', (fpr[idx]*100, tpr[idx]*100),
                    textcoords="offset points", xytext=(10, -5), fontsize=9, color=color)

    ax.set_xlabel('False Acceptance Rate — FAR (%)', fontsize=12)
    ax.set_ylabel('True Acceptance Rate — TAR (%)', fontsize=12)
    ax.set_title(f'ROC Curve — SCA-MobileNet (3.19M params)', fontsize=13)
    ax.set_xlim(-0.2, 5)
    ax.set_ylim(95, 100.2)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    textstr = (f'EER    : {eer:.4f}%\n'
               f'AUC    : {roc_auc:.6f}\n'
               f'TAR@0.01%FAR : {tar_001:.2f}%\n'
               f'TAR@0.1%FAR  : {tar_01:.2f}%\n'
               f'TAR@1%FAR    : {tar_1:.2f}%')
    props = dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.85)
    ax.text(0.98, 0.35, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=props, fontfamily='monospace')

    plt.tight_layout()
    roc_path = PAPER_DIR / 'fig_roc.png'
    plt.savefig(str(roc_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {roc_path}")

    # ========== DET Curve ==========
    thresholds = np.linspace(min(genuine.min(), impostor.min()),
                             max(genuine.max(), impostor.max()), 2000)
    far_vals = np.array([np.mean(impostor >= t) for t in thresholds])
    frr_vals = np.array([np.mean(genuine < t) for t in thresholds])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Probit scale
    ax1.plot(far_vals * 100, frr_vals * 100, 'b-', linewidth=2)
    ax1.plot([0, 50], [0, 50], 'k--', alpha=0.4, label='EER Line')
    eer_idx = np.argmin(np.abs(far_vals - frr_vals))
    ax1.plot(far_vals[eer_idx]*100, frr_vals[eer_idx]*100, 'g*', markersize=15,
             label=f'EER = {eer:.2f}%', zorder=5)
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11)
    ax1.set_ylabel('False Negative Rate (FNR)', fontsize=11)
    ax1.set_title('(a) DET Curve (Probit Scale)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, 50)

    # (b) Log scale
    mask = (far_vals > 0) & (frr_vals > 0)
    ax2.plot(far_vals[mask] * 100, frr_vals[mask] * 100, 'b-', linewidth=2, label='SCA-MobileNet')
    ax2.plot(far_vals[eer_idx]*100, frr_vals[eer_idx]*100, 'g*', markersize=15,
             label=f'EER = {eer:.2f}%', zorder=5)

    for far_t, tar_v, color, name in [
        (0.0001, tar_001, '#E65100', f'FAR=0.01%: FNR={100-tar_001:.2f}%'),
        (0.001, tar_01, '#7B1FA2', f'FAR=0.1%: FNR={100-tar_01:.2f}%'),
        (0.01, tar_1, '#1565C0', f'FAR=1%: FNR={100-tar_1:.2f}%'),
    ]:
        idx = np.argmin(np.abs(far_vals - far_t))
        if far_vals[idx] > 0 and frr_vals[idx] > 0:
            ax2.plot(far_vals[idx]*100, frr_vals[idx]*100, 'o', color=color, markersize=8, zorder=5, label=name)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('False Positive Rate (%)', fontsize=11)
    ax2.set_ylabel('False Negative Rate (%)', fontsize=11)
    ax2.set_title('(b) DET Curve (Log Scale)', fontsize=12)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')

    plt.suptitle('Detection Error Tradeoff (DET) Curve Analysis', fontsize=14, y=1.02)

    info = f'Model: SCA-MobileNet 3M (Bottleneck) | Loss: AdaCos | Best Epoch: 40\nEER: {eer:.2f}% | TAR@0.01%FAR: {tar_001:.2f}%'
    fig.text(0.5, -0.02, info, ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    plt.tight_layout()
    det_path = PAPER_DIR / 'fig_det.png'
    plt.savefig(str(det_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {det_path}")

    # ========== Score Distribution (real data) ==========
    fig, ax = plt.subplots(figsize=(12, 5))
    bins = np.linspace(0, 1, 100)
    ax.hist(genuine, bins, alpha=0.7, color='#43A047', label='Genuine pairs', density=True)
    ax.hist(impostor, bins, alpha=0.7, color='#E53935', label='Impostor pairs', density=True)

    eer_thr = results.get('EER_threshold', thresholds[eer_idx])
    ax.axvline(x=eer_thr, color='#1565C0', linestyle='--', linewidth=2,
               label=f'EER threshold ≈ {eer_thr:.2f}')

    ax.set_xlabel('Cosine Similarity Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Score Distribution: Genuine vs Impostor (SCA-MobileNet, Internal Dataset)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.annotate(f'd-prime = {d_prime:.2f}', xy=(0.65, ax.get_ylim()[1] * 0.8),
                fontsize=11, fontweight='bold', color='#1565C0',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    score_path = PAPER_DIR / 'fig_score_distribution.png'
    plt.savefig(str(score_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {score_path}")


if __name__ == '__main__':
    main()
