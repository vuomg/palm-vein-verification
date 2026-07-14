"""
Enrollment-Probe (5:5) Evaluation Protocol for Palm Vein Authentication.

Protocol:
  - Each test identity has 10 images
  - 5 images → enrollment (template = mean of L2-normalized embeddings)
  - 5 images → probe (compare against ALL templates)
  - Genuine: probe vs own template
  - Impostor: probe vs every other template
  - Metrics: EER, TAR@FAR=0.01%, TAR@FAR=0.1%, TAR@FAR=1%, AUC, D-prime

Usage:
  python evaluation/enrollment_probe_eval.py --model sca_mobilenet --dataset SCUT
  python evaluation/enrollment_probe_eval.py --model rsnet --dataset TONGJI
  python evaluation/enrollment_probe_eval.py --all
"""

import os
import sys
import json
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'models'))

# ──────────────────────────────────────────────────────────────────
# Configuration: checkpoint map and model/dataset metadata
# ──────────────────────────────────────────────────────────────────

RESULTS_DIR = PROJECT_ROOT / 'results'

CHECKPOINT_MAP = {
    ('sca_mobilenet', 'SCUT'):     RESULTS_DIR / 'results_scut_sca_mobilenet_sca_mobilenet' / 'best_sca_mobilenet_model_eer.pth',
    ('sca_mobilenet', 'VERA'):     RESULTS_DIR / 'results_vera_sca_mobilenet_sca_mobilenet' / 'best_sca_mobilenet_model_eer.pth',
    ('sca_mobilenet', 'TONGJI'):   RESULTS_DIR / 'results_tongji_sca_sca_mobilenet_sca_mobilenet' / 'best_sca_mobilenet_model_eer.pth',
    ('sca_mobilenet', 'HUTECH_CNPV1549'): RESULTS_DIR / 'results_sca_v2_sca_mobilenet' / 'best_sca_mobilenet_model_eer.pth',

    ('rsnet', 'SCUT'):     RESULTS_DIR / 'results_scut_rsnet_rsnet' / 'best_rsnet_model_eer.pth',
    ('rsnet', 'VERA'):     RESULTS_DIR / 'results_vera_rsnet_rsnet' / 'best_rsnet_model_eer.pth',
    ('rsnet', 'TONGJI'):   RESULTS_DIR / 'results_tongji_rsnet_rsnet' / 'best_rsnet_model_eer.pth',
    ('rsnet', 'HUTECH_CNPV1549'): RESULTS_DIR / 'results_hutech_cnpv1549' / 'rsnet_rsnet' / 'best_rsnet_model_eer.pth',

    ('fgfnet', 'SCUT'):     RESULTS_DIR / 'results_scut_fgfnet_fgfnet' / 'best_fgfnet_model_eer.pth',
    ('fgfnet', 'VERA'):     RESULTS_DIR / 'results_vera_fgfnet_fgfnet' / 'best_fgfnet_model_eer.pth',
    ('fgfnet', 'TONGJI'):   RESULTS_DIR / 'results_tongji_fgfnet_fgfnet' / 'best_fgfnet_model_eer.pth',

    ('mpsnet', 'SCUT'):     RESULTS_DIR / 'results_scut_mpsnet_mpsnet' / 'best_mspnet_model_eer.pth',
    ('mpsnet', 'VERA'):     RESULTS_DIR / 'results_vera_mpsnet_mpsnet' / 'best_mspnet_model_eer.pth',
    ('mpsnet', 'TONGJI'):   RESULTS_DIR / 'results_tongji_mpsnet_mpsnet' / 'best_mspnet_model_eer.pth',
    ('mpsnet', 'HUTECH_CNPV1549'): RESULTS_DIR / 'results_hutech_cnpv1549' / 'mpsnet_mpsnet' / 'best_mpsnet_model_eer.pth',

    ('gscl', 'TONGJI'):    RESULTS_DIR / 'results_tongji_gscl_gscl' / 'best_gscl_model_eer.pth',
    ('gscl', 'HUTECH_CNPV1549'):  RESULTS_DIR / 'results_hutech_cnpv1549' / 'resnet50_gscl' / 'best_gscl_model_eer.pth',

    ('eusipco2020', 'SCUT'):   RESULTS_DIR / 'results_scut_densenet161_eusipco2020' / 'best_eusipco2020_model_eer.pth',
    ('eusipco2020', 'VERA'):   RESULTS_DIR / 'results_vera_densenet161_eusipco2020' / 'best_eusipco2020_model_eer.pth',
    ('eusipco2020', 'TONGJI'): RESULTS_DIR / 'results_tongji_densenet161_eusipco2020' / 'best_eusipco2020_model_eer.pth',
}

DATASET_MAP = {
    'SCUT':     PROJECT_ROOT / 'datasets' / 'SCUT_dataset_openset' / 'test',
    'VERA':     PROJECT_ROOT / 'datasets' / 'VERA_dataset_openset' / 'test',
    'TONGJI':   PROJECT_ROOT / 'datasets' / 'TONGJI_dataset_openset' / 'test',
    'HUTECH_CNPV1549': PROJECT_ROOT / 'datasets' / 'final_dataset_openset' / 'test',
}

MODEL_INPUT_SIZE = {
    'sca_mobilenet': 224,
    'rsnet': 224,
    'fgfnet': 256,
    'mpsnet': 224,
    'gscl': 224,
    'eusipco2020': 224,
}

MODEL_IMAGE_MODE = {
    'sca_mobilenet': 'RGB',
    'rsnet': 'RGB',
    'fgfnet': 'RGB',
    'mpsnet': 'L',
    'gscl': 'RGB',
    'eusipco2020': 'RGB',
}


# ──────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────

class PalmVeinTestDataset(Dataset):
    """Load test images preserving per-identity order for 5:5 split."""

    def __init__(self, data_dir, transform=None, image_mode='RGB'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_mode = image_mode
        self.samples = []          # (path, label_idx)
        self.class_to_idx = {}
        self.identity_images = {}  # label_idx → [paths in sorted order]

        class_idx = 0
        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            self.class_to_idx[class_dir.name] = class_idx
            imgs = sorted(
                list(class_dir.glob('*.png'))
                + list(class_dir.glob('*.jpg'))
                + list(class_dir.glob('*.bmp'))
            )
            self.identity_images[class_idx] = [str(p) for p in imgs]
            for p in imgs:
                self.samples.append((str(p), class_idx))
            class_idx += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert(self.image_mode)
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────

def load_model(model_name, checkpoint_path, device):
    """Instantiate model architecture and load weights from checkpoint."""
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    feature_dim = config.get('feature_dim', 1024)
    num_classes = config.get('num_classes', 100)

    if model_name == 'rsnet':
        from RSNet.model import RSNet
        model = RSNet(
            feature_dim=feature_dim,
            in_channels=3,
            dropout_rate=config.get('dropout', 0.3),
            d=config.get('d', 2),
            n_groups=config.get('n_groups', 4),
        )

    elif model_name == 'sca_mobilenet':
        from SCA_MobileNet.model import SCAMobileNet
        sd = ckpt['model_state_dict']
        use_bottleneck = config.get('use_bottleneck', 'channel_bottleneck.0.weight' in sd)
        ca_reduction = config.get('ca_reduction', None)
        if ca_reduction is None:
            ca_key = 'scib.coord_att.conv1.weight'
            if ca_key in sd:
                mid_ch = sd[ca_key].shape[0]
                in_ch = sd[ca_key].shape[1]
                ca_reduction = in_ch // mid_ch
            else:
                ca_reduction = 8
        model = SCAMobileNet(
            embedding_size=feature_dim,
            class_size=num_classes,
            pretrained=False,
            only_embeddings=True,
            use_stn=config.get('use_stn', True),
            use_ca=config.get('use_ca', True),
            use_spp=config.get('use_spp', True),
            dropout=config.get('dropout', 0.3),
            use_bottleneck=use_bottleneck,
            ca_reduction=ca_reduction,
        )

    elif model_name == 'fgfnet':
        from FGFNet.model import MobileViT_FFC_ATTN_FFTSA
        model = MobileViT_FFC_ATTN_FFTSA(
            image_size=(256, 256),
            num_classes=num_classes,
        )

    elif model_name == 'mpsnet':
        from MPSNet_2022.model_pytorch import MPSNet
        model = MPSNet(
            feature_dim=feature_dim,
            input_channels=1,
            dropout=config.get('dropout', 0.3),
        )

    elif model_name == 'gscl':
        gscl_path = str(PROJECT_ROOT / 'models' / 'GSCL-PyTorch' / 'vein_feature_learning')
        if gscl_path not in sys.path:
            sys.path.insert(0, gscl_path)
        from models.models import ResNets
        backbone = config.get('backbone', config.get('gscl_backbone', 'resnet50'))
        model = ResNets(
            backbone=backbone,
            head_type='cls_norm',
            num_classes=num_classes,
        )

    elif model_name == 'eusipco2020':
        eusipco_dir = PROJECT_ROOT / 'models' / 'Modified_Densenet161_2021'
        spec = importlib.util.spec_from_file_location(
            'modified_models',
            eusipco_dir / 'models' / 'modified_models.py',
        )
        modified_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modified_models)
        DenseNet161_Modified = modified_models.DenseNet161_Modified
        model = DenseNet161_Modified(
            embedding_size=feature_dim,
            class_size=num_classes,
            pretrained=False,
            only_embeddings=True,
            l2_normed=True,
        )

    else:
        raise ValueError(f'Unknown model: {model_name}')

    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if unexpected:
        print(f'  (ignored {len(unexpected)} unexpected keys: {unexpected[:3]})')
    if missing:
        print(f'  WARNING: {len(missing)} missing keys: {missing[:3]}')
    model = model.to(device)
    model.eval()
    print(f'  Loaded {model_name} from {checkpoint_path.name} (epoch {ckpt.get("epoch", "?")})')
    return model


# ──────────────────────────────────────────────────────────────────
# Embedding extraction
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, dataloader, device, model_name):
    """Extract L2-normalized embeddings."""
    model.eval()
    all_emb, all_lbl = [], []

    for images, labels in tqdm(dataloader, desc='  Extracting', leave=False):
        images = images.to(device, non_blocking=True)

        if model_name == 'eusipco2020':
            emb = model(images, train=False)
        elif model_name == 'gscl':
            emb, _ = model(images)
        elif model_name == 'fgfnet':
            emb = model.get_embedding(images)
        elif model_name == 'veinkan':
            emb = model.get_embedding(images)
        else:
            out = model(images)
            emb = out[0] if isinstance(out, tuple) else out

        emb = F.normalize(emb, p=2, dim=1)
        all_emb.append(emb.cpu().numpy())
        all_lbl.append(labels.numpy())

    return np.concatenate(all_emb), np.concatenate(all_lbl)


# ──────────────────────────────────────────────────────────────────
# Enrollment-Probe (5:5) evaluation core
# ──────────────────────────────────────────────────────────────────

def enrollment_probe_eval(embeddings, labels, n_enroll=5, n_probe=5, seed=42):
    """
    Run the enrollment-probe protocol.

    For each identity (≥10 images):
      - First 5 (sorted) → enrollment, template = mean of L2-normed embeddings
      - Last 5 → probe
    Genuine: each probe vs its own template
    Impostor: each probe vs every other template

    Returns dict with EER, TAR@FAR, AUC, D-prime, score stats.
    """
    np.random.seed(seed)
    unique_labels = np.unique(labels)

    templates = {}     # label → template vector
    probe_embs = []    # list of (embedding, label)

    skipped = 0
    for lbl in unique_labels:
        idxs = np.where(labels == lbl)[0]
        if len(idxs) < n_enroll + n_probe:
            skipped += 1
            continue

        enroll_idx = idxs[:n_enroll]
        probe_idx = idxs[n_enroll:n_enroll + n_probe]

        enroll_vecs = embeddings[enroll_idx]
        enroll_vecs = enroll_vecs / (np.linalg.norm(enroll_vecs, axis=1, keepdims=True) + 1e-8)
        template = enroll_vecs.mean(axis=0)
        template = template / (np.linalg.norm(template) + 1e-8)
        templates[lbl] = template

        for idx in probe_idx:
            probe_embs.append((embeddings[idx], lbl))

    n_ids = len(templates)
    n_probes = len(probe_embs)
    if skipped > 0:
        print(f'  Skipped {skipped} identities with <{n_enroll + n_probe} images')
    print(f'  Enrolled {n_ids} identities, {n_probes} probe samples')

    template_labels = sorted(templates.keys())
    template_matrix = np.stack([templates[l] for l in template_labels])

    genuine_scores = []
    impostor_scores = []

    for emb, lbl in probe_embs:
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        sims = template_matrix @ emb_norm  # cosine sim (templates already normalized)

        lbl_pos = template_labels.index(lbl)
        genuine_scores.append(sims[lbl_pos])

        imp_mask = np.ones(len(template_labels), dtype=bool)
        imp_mask[lbl_pos] = False
        impostor_scores.extend(sims[imp_mask].tolist())

    genuine_scores = np.array(genuine_scores, dtype=np.float64)
    impostor_scores = np.array(impostor_scores, dtype=np.float64)

    results = compute_metrics(genuine_scores, impostor_scores)
    results['n_identities'] = n_ids
    results['n_probes'] = n_probes
    results['n_genuine_scores'] = len(genuine_scores)
    results['n_impostor_scores'] = len(impostor_scores)
    return results, genuine_scores, impostor_scores


def compute_metrics(genuine, impostor):
    """Compute EER, TAR@FAR, AUC, D-prime from score arrays."""
    y_true = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    y_scores = np.concatenate([genuine, impostor])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fnr[eer_idx] + fpr[eer_idx]) / 2)
    eer_threshold = float(thresholds[eer_idx])

    auc = float(roc_auc_score(y_true, y_scores))

    def tar_at_far(target_far):
        thresh = np.percentile(impostor, 100 * (1 - target_far))
        return float(np.mean(genuine >= thresh))

    tar_001 = tar_at_far(0.0001)
    tar_01 = tar_at_far(0.001)
    tar_1 = tar_at_far(0.01)

    g_mean, g_std = float(np.mean(genuine)), float(np.std(genuine))
    i_mean, i_std = float(np.mean(impostor)), float(np.std(impostor))
    d_prime = abs(g_mean - i_mean) / (np.sqrt(0.5 * (g_std**2 + i_std**2)) + 1e-8)

    return {
        'EER': eer,
        'EER_threshold': eer_threshold,
        'TAR@0.01%FAR': tar_001,
        'TAR@0.1%FAR': tar_01,
        'TAR@1%FAR': tar_1,
        'AUC': auc,
        'd_prime': float(d_prime),
        'genuine_mean': g_mean,
        'genuine_std': g_std,
        'impostor_mean': i_mean,
        'impostor_std': i_std,
    }


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def run_single(model_name, dataset_name, device, output_dir):
    """Evaluate one model on one dataset."""
    key = (model_name, dataset_name)
    ckpt_path = CHECKPOINT_MAP.get(key)
    if ckpt_path is None or not ckpt_path.exists():
        print(f'SKIP {model_name}/{dataset_name}: no checkpoint')
        return None

    data_dir = DATASET_MAP.get(dataset_name)
    if data_dir is None or not data_dir.exists():
        print(f'SKIP {model_name}/{dataset_name}: dataset not found at {data_dir}')
        return None

    print(f'\n{"="*60}')
    print(f'  {model_name} on {dataset_name}')
    print(f'{"="*60}')

    img_size = MODEL_INPUT_SIZE[model_name]
    img_mode = MODEL_IMAGE_MODE[model_name]

    if img_mode == 'L':
        tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    else:
        tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    dataset = PalmVeinTestDataset(data_dir, transform=tfm, image_mode=img_mode)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    model = load_model(model_name, ckpt_path, device)
    embeddings, labels = extract_embeddings(model, loader, device, model_name)
    print(f'  Extracted {len(embeddings)} embeddings, {len(np.unique(labels))} identities')

    results, genuine, impostor = enrollment_probe_eval(embeddings, labels)

    print(f'\n  Results:')
    print(f'    EER:            {results["EER"]*100:.4f}%')
    print(f'    TAR@0.01%FAR:   {results["TAR@0.01%FAR"]*100:.2f}%')
    print(f'    TAR@0.1%FAR:    {results["TAR@0.1%FAR"]*100:.2f}%')
    print(f'    TAR@1%FAR:      {results["TAR@1%FAR"]*100:.2f}%')
    print(f'    AUC:            {results["AUC"]:.6f}')
    print(f'    D-prime:        {results["d_prime"]:.4f}')

    result_file = output_dir / f'{model_name}_{dataset_name}.json'
    results['model'] = model_name
    results['dataset'] = dataset_name
    results['checkpoint'] = str(ckpt_path.relative_to(PROJECT_ROOT))
    results['timestamp'] = datetime.now().isoformat()
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Saved to {result_file.name}')

    del model
    torch.cuda.empty_cache()
    return results


def run_all(device, output_dir):
    """Run evaluation for all available model-dataset combos."""
    all_results = []
    for (model_name, dataset_name), ckpt in CHECKPOINT_MAP.items():
        r = run_single(model_name, dataset_name, device, output_dir)
        if r is not None:
            all_results.append(r)

    if all_results:
        summary_path = output_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\nSummary saved to {summary_path}')
        print_summary_table(all_results)


def print_summary_table(results_list):
    """Print a formatted summary table."""
    print(f'\n{"="*90}')
    print(f'  ENROLLMENT-PROBE (5:5) EVALUATION SUMMARY')
    print(f'{"="*90}')
    header = f'{"Model":<16} {"Dataset":<10} {"EER%":>8} {"TAR@0.01%":>10} {"TAR@0.1%":>10} {"TAR@1%":>8} {"D-prime":>8}'
    print(header)
    print('-' * 90)
    for r in results_list:
        print(f'{r["model"]:<16} {r["dataset"]:<10} '
              f'{r["EER"]*100:>7.3f}% '
              f'{r["TAR@0.01%FAR"]*100:>9.2f}% '
              f'{r["TAR@0.1%FAR"]*100:>9.2f}% '
              f'{r["TAR@1%FAR"]*100:>7.2f}% '
              f'{r["d_prime"]:>8.3f}')
    print(f'{"="*90}')


def main():
    parser = argparse.ArgumentParser(description='Enrollment-Probe (5:5) Evaluation')
    parser.add_argument('--model', type=str, default=None,
                        choices=['sca_mobilenet', 'rsnet', 'fgfnet', 'mpsnet', 'gscl', 'eusipco2020'],
                        help='Model to evaluate (omit for --all)')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['SCUT', 'VERA', 'TONGJI', 'HUTECH_CNPV1549'],
                        help='Dataset to evaluate on')
    parser.add_argument('--all', action='store_true', help='Evaluate all available combos')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    print(f'Device: {device}')

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / 'results_enroll_probe'
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all or (args.model is None and args.dataset is None):
        run_all(device, output_dir)
    elif args.model and args.dataset:
        run_single(args.model, args.dataset, device, output_dir)
    elif args.model:
        all_results = []
        for ds in ['SCUT', 'VERA', 'TONGJI', 'HUTECH_CNPV1549']:
            r = run_single(args.model, ds, device, output_dir)
            if r is not None:
                all_results.append(r)
        if all_results:
            print_summary_table(all_results)
    elif args.dataset:
        all_results = []
        for m in ['sca_mobilenet', 'rsnet', 'fgfnet', 'mpsnet', 'gscl', 'eusipco2020']:
            r = run_single(m, args.dataset, device, output_dir)
            if r is not None:
                all_results.append(r)
        if all_results:
            print_summary_table(all_results)


if __name__ == '__main__':
    main()
