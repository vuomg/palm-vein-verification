"""
Statistical rigor metrics for SCA-MobileNet on Tongji 50:50 (same protocol/test
set as verify_results.py): Bootstrap 95% CI for EER, D-prime (decidability),
and a DET curve. Caches embeddings to .npy so re-runs are fast.
"""
import os, sys, itertools
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

ROOT = os.path.dirname(__file__)
TONGJI = os.path.join(ROOT, 'datasets', 'TONGJI_Dataset', 'ROI')
S1, S2 = os.path.join(TONGJI, 'session1'), os.path.join(TONGJI, 'session2')
CKPT = os.path.join(ROOT, 'thesis', 'tongji_benchmark',
                    'results_adaface_12m_cosine', 'checkpoints', 'best_model.pth')
CHARTS = os.path.join(ROOT, 'thesis', 'charts')
CACHE = os.path.join(ROOT, 'thesis', 'tongji_benchmark',
                     'results_adaface_12m_cosine', 'test_embeddings.npy')
os.makedirs(CHARTS, exist_ok=True)

N_SUBJECTS, N_TRAIN, N_IMG = 600, 300, 20

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.35,
                     "axes.spines.top": False, "axes.spines.right": False})

# ── Build test subject list (palms 301-600), same as benchmark ──
subjects = []
for subj in range(N_SUBJECTS):
    start = subj * 10 + 1
    imgs = [os.path.join(sess, f'{i:05d}.bmp')
            for sess in (S1, S2) for i in range(start, start + 10)]
    subjects.append(imgs)
test_subjects = subjects[N_TRAIN:]

# ── Extract (or load cached) embeddings ──────────────────────
if os.path.exists(CACHE):
    print(f"Loading cached embeddings: {CACHE}")
    data = np.load(CACHE)
    embs, labels = data[:, :-1].astype(np.float32), data[:, -1].astype(int)
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    print(f"Checkpoint epoch {ckpt['epoch']}, EER {ckpt['eer']:.4f}%")
    from SCA_MobileNet.model import SCAMobileNet
    model = SCAMobileNet(embedding_size=1024, only_embeddings=True,
                         use_bottleneck=False, ca_reduction=32).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.5]*3, [0.5]*3)])
    embs, labels = [], []
    paths = [(lbl, p) for lbl, imgs in enumerate(test_subjects) for p in imgs]
    BATCH = 64
    print(f"Extracting {len(paths)} embeddings (batched)...")
    for i in tqdm(range(0, len(paths), BATCH)):
        chunk = paths[i:i+BATCH]
        batch = torch.stack([tf(Image.open(p).convert('RGB')) for _, p in chunk]).to(device)
        with torch.no_grad():
            out = model(batch)
            e = out[0] if isinstance(out, tuple) else out
            e = F.normalize(e, dim=1).cpu().numpy()
        embs.append(e)
        labels.extend(l for l, _ in chunk)
    embs = np.concatenate(embs).astype(np.float32)
    labels = np.array(labels)
    np.save(CACHE, np.concatenate([embs, labels[:, None]], axis=1))
    print(f"Cached embeddings → {CACHE}")

print(f"Embeddings: {embs.shape}")

# ── Generate pairs (same as verify_results.py) ───────────────
genuine = []
for lbl in range(len(test_subjects)):
    base = lbl * N_IMG
    for a, b in itertools.combinations(range(N_IMG), 2):
        genuine.append((base+a, base+b))
rng = np.random.RandomState(42)
n = len(embs)
impostor = []
while len(impostor) < len(genuine):
    i, j = rng.choice(n, 2, replace=False)
    if labels[i] != labels[j]:
        impostor.append((int(i), int(j)))

g_scores = np.array([embs[i] @ embs[j] for i, j in genuine])
i_scores = np.array([embs[i] @ embs[j] for i, j in impostor])
scores = np.concatenate([g_scores, i_scores])
y = np.concatenate([np.ones(len(g_scores)), np.zeros(len(i_scores))])
print(f"Pairs: {len(g_scores)} genuine + {len(i_scores)} impostor")


def compute_eer(yv, sv):
    fpr, tpr, thr = roc_curve(yv, sv)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2 * 100, thr[idx], fpr, fnr, tpr


eer, thr_eer, fpr, fnr, tpr = compute_eer(y, scores)
roc_auc = auc(fpr, tpr)

# ── D-prime (decidability index) ─────────────────────────────
dprime = abs(g_scores.mean() - i_scores.mean()) / np.sqrt(
    0.5 * (g_scores.var() + i_scores.var()))

# ── Bootstrap 95% CI for EER (resample pairs) ────────────────
B = 1000
rng2 = np.random.RandomState(0)
ng, ni = len(g_scores), len(i_scores)
boot = np.empty(B)
for b in range(B):
    gi = rng2.randint(0, ng, ng)
    ii = rng2.randint(0, ni, ni)
    sv = np.concatenate([g_scores[gi], i_scores[ii]])
    yv = np.concatenate([np.ones(ng), np.zeros(ni)])
    boot[b] = compute_eer(yv, sv)[0]
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

print("\n" + "="*52)
print(f"EER            : {eer:.4f}%")
print(f"EER 95% CI     : [{ci_lo:.4f}%, {ci_hi:.4f}%]  (bootstrap B={B})")
print(f"D-prime        : {dprime:.4f}")
print(f"AUC            : {roc_auc:.6f}")
print(f"Genuine  mean  : {g_scores.mean():.4f}  std {g_scores.std():.4f}")
print(f"Impostor mean  : {i_scores.mean():.4f}  std {i_scores.std():.4f}")
print("="*52)

# ── DET curve (FMR vs FNMR, log-log) ─────────────────────────
fmr, tprc, _ = roc_curve(y, scores)
fnmr = 1 - tprc
m = (fmr > 0) & (fnmr > 0)
fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.plot(fmr[m]*100, fnmr[m]*100, color='#2563EB', lw=2.5,
        label=f'SCA-MobileNet (EER={eer:.3f}%)')
ax.plot([1e-3, 100], [1e-3, 100], '--', color='#9CA3AF', lw=1.2, label='FMR = FNMR')
ax.plot(eer, eer, 'go', ms=10, label=f'EER = {eer:.3f}%  [CI: {ci_lo:.3f}–{ci_hi:.3f}%]')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim([1e-3, 100]); ax.set_ylim([1e-3, 100])
ax.set_xlabel('FMR — False Match Rate (%)', fontsize=12)
ax.set_ylabel('FNMR — False Non-Match Rate (%)', fontsize=12)
ax.legend(fontsize=9.5, loc='upper right')
ax.set_title(f'Hình 4.6: DET Curve — SCA-MobileNet, Tongji 50:50\n'
             f'EER={eer:.4f}% (95% CI {ci_lo:.3f}–{ci_hi:.3f}%) | D-prime={dprime:.3f}',
             fontsize=11, pad=10)
plt.tight_layout()
det_path = os.path.join(CHARTS, 'fig_4_6_det_curve.png')
plt.savefig(det_path, bbox_inches='tight'); plt.close()
print(f"\nDET curve saved: {det_path}")

# ── Save metrics for thesis ──────────────────────────────────
import json
out = {"eer": float(eer), "eer_ci95": [float(ci_lo), float(ci_hi)],
       "dprime": float(dprime), "auc": float(roc_auc),
       "genuine_mean": float(g_scores.mean()), "genuine_std": float(g_scores.std()),
       "impostor_mean": float(i_scores.mean()), "impostor_std": float(i_scores.std()),
       "bootstrap_B": B}
with open(os.path.join(ROOT, 'thesis', 'tongji_benchmark',
          'results_adaface_12m_cosine', 'rigor_metrics.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("Saved rigor_metrics.json")
