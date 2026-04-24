"""
Cross-Domain Evaluation: VERA <-> SCUT
=======================================
Direction 1: Train VERA -> Test SCUT (entire dataset)
Direction 2: Train SCUT -> Test VERA (entire dataset)

Usage:
    python evaluation/cross_domain_vera_scut.py --direction vera_to_scut --model sca_mobilenet
    python evaluation/cross_domain_vera_scut.py --direction scut_to_vera --model sca_mobilenet
    python evaluation/cross_domain_vera_scut.py --direction vera_to_scut --all
    python evaluation/cross_domain_vera_scut.py --direction scut_to_vera --all
"""

import sys
import json
import argparse
import importlib
import importlib.util
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

try:
    import timm
except ImportError:
    timm = None

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'models'))

from biometric.metrics import BiometricEvaluator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VERA_DIR = project_root / "datasets" / "VERA_dataset_openset"
SCUT_DIR = project_root / "datasets" / "SCUT_dataset_openset"
OUTPUT_DIR = project_root / "results" / "results_cross_domain_vera_scut"

# =====================================================================
# Checkpoint configs per source dataset
# =====================================================================
VERA_TRAINED = {
    'sca_mobilenet': {
        'name': 'SCA-MobileNet',
        'checkpoint': 'results/results_vera_sca_mobilenet_sca_mobilenet/best_sca_mobilenet_model_eer.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'sca_mobilenet',
        'num_train_classes': 154,
    },
    'mpsnet': {
        'name': 'MPSNet',
        'checkpoint': 'results/results_vera_mpsnet_mpsnet/best_mpsnet_model_eer.pth',
        'input_channels': 1, 'image_size': 224, 'model_type': 'mpsnet',
        'num_train_classes': 154,
    },
    'eusipco2020': {
        'name': 'Modified-DenseNet161',
        'checkpoint': 'results/results_vera_densenet161_eusipco2020/best_eusipco2020_model_eer.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'eusipco2020',
        'num_train_classes': 154,
    },
    'rsnet': {
        'name': 'RSNet',
        'checkpoint': 'results/results_vera_rsnet_rsnet/best_rsnet_model_eer.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'rsnet',
        'num_train_classes': 154,
    },
    'fgfnet': {
        'name': 'FGFNet',
        'checkpoint': 'results/results_vera_fgfnet_fgfnet/best_fgfnet_model_eer.pth',
        'input_channels': 3, 'image_size': 256, 'model_type': 'fgfnet',
        'num_train_classes': 154,
    },
    'mobilenetv3_base': {
        'name': 'MobileNetV3-Base',
        'checkpoint': 'results/results_vera_mobilenetv3_base_sca_mobilenet/best_sca_mobilenet_model_eer.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'mobilenetv3_base',
        'num_train_classes': 154,
    },
    'efficientnet_b0': {
        'name': 'EfficientNet-B0',
        'checkpoint': 'results/results_vera_efficientnet_b0_EfficientNet-B0/best_EfficientNet-B0_model_eer.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'sca_transformer',
        'backbone_name': 'efficientnet_b0',
        'num_train_classes': 154,
    },
    'deit_tiny': {
        'name': 'DeiT-Tiny',
        'checkpoint': 'results/results_vera_deit_tiny_DeiT-Tiny/best_DeiT-Tiny_model_eer.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'sca_transformer',
        'backbone_name': 'deit_tiny',
        'num_train_classes': 154,
    },
    'swin_tiny': {
        'name': 'Swin-Tiny',
        'checkpoint': 'results/results_vera_swin_tiny_Swin-Tiny/best_Swin-Tiny_model_eer.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'sca_transformer',
        'backbone_name': 'swin_tiny',
        'num_train_classes': 154,
    },
    'mobilevit_s': {
        'name': 'MobileViT-S',
        'checkpoint': 'results/results_vera_mobilevit_s_MobileViT-S/best_MobileViT-S_model_eer.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'sca_transformer',
        'backbone_name': 'mobilevit_s',
        'num_train_classes': 154,
    },
    'gscl': {
        'name': 'GSCL (ResNet18)',
        'checkpoint': 'models/GSCL-PyTorch/vein_feature_learning/results/vera_resnet18_resnet18/checkpoints/best_model_seed42.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'gscl',
        'backbone_name': 'resnet18',
        'num_train_classes': 154,
    },
    'resnet50': {
        'name': 'ResNet50',
        'checkpoint': 'models/GSCL-PyTorch/vein_feature_learning/results/vera_resnet50_resnet50/checkpoints/best_model_seed42.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'gscl',
        'backbone_name': 'resnet50',
        'num_train_classes': 154,
    },
}

SCUT_TRAINED = {
    'sca_mobilenet': {
        'name': 'SCA-MobileNet',
        'checkpoint': 'results/results_scut_sca_mobilenet_sca_mobilenet/best_sca_mobilenet_model_eer.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'sca_mobilenet',
        'num_train_classes': 770,
    },
    'mpsnet': {
        'name': 'MPSNet',
        'checkpoint': 'results/results_scut_mpsnet_mpsnet/best_mpsnet_model_eer.pth',
        'input_channels': 1, 'image_size': 224, 'model_type': 'mpsnet',
        'num_train_classes': 770,
    },
    'eusipco2020': {
        'name': 'Modified-DenseNet161',
        'checkpoint': 'results/results_scut_densenet161_eusipco2020/best_eusipco2020_model_eer.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'eusipco2020',
        'num_train_classes': 770,
    },
    'rsnet': {
        'name': 'RSNet',
        'checkpoint': 'results/results_scut_rsnet_rsnet/best_rsnet_model_eer.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'rsnet',
        'num_train_classes': 770,
    },
    'fgfnet': {
        'name': 'FGFNet',
        'checkpoint': 'results/results_scut_fgfnet_fgfnet/best_fgfnet_model_eer.pth',
        'input_channels': 3, 'image_size': 256, 'model_type': 'fgfnet',
        'num_train_classes': 770,
    },
    'mobilenetv3_base': {
        'name': 'MobileNetV3-Base',
        'checkpoint': 'results/results_scut_mobilenetv3_base_sca_mobilenet/best_sca_mobilenet_model_eer.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'mobilenetv3_base',
        'num_train_classes': 770,
    },
    'efficientnet_b0': {
        'name': 'EfficientNet-B0',
        'checkpoint': 'results/results_scut_efficientnet_b0_EfficientNet-B0/best_EfficientNet-B0_model_eer.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'sca_transformer',
        'backbone_name': 'efficientnet_b0',
        'num_train_classes': 770,
    },
    'deit_tiny': {
        'name': 'DeiT-Tiny',
        'checkpoint': 'results/results_scut_deit_tiny_DeiT-Tiny/best_DeiT-Tiny_model_eer.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'sca_transformer',
        'backbone_name': 'deit_tiny',
        'num_train_classes': 770,
    },
    'swin_tiny': {
        'name': 'Swin-Tiny',
        'checkpoint': 'results/results_scut_swin_tiny_Swin-Tiny/best_Swin-Tiny_model_eer.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'sca_transformer',
        'backbone_name': 'swin_tiny',
        'num_train_classes': 770,
    },
    'mobilevit_s': {
        'name': 'MobileViT-S',
        'checkpoint': 'results/results_scut_mobilevit_s_MobileViT-S/best_MobileViT-S_model_eer.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'sca_transformer',
        'backbone_name': 'mobilevit_s',
        'num_train_classes': 770,
    },
    'gscl': {
        'name': 'GSCL (ResNet18)',
        'checkpoint': 'models/GSCL-PyTorch/vein_feature_learning/results/scut_resnet18_resnet18/checkpoints/best_model_seed42.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'gscl',
        'backbone_name': 'resnet18',
        'num_train_classes': 770,
    },
    'resnet50': {
        'name': 'ResNet50',
        'checkpoint': 'models/GSCL-PyTorch/vein_feature_learning/results/scut_resnet50_resnet50/checkpoints/best_model_seed42.pth',
        'input_channels': 3, 'image_size': 224, 'model_type': 'gscl',
        'backbone_name': 'resnet50',
        'num_train_classes': 770,
    },
}

SOURCE_CONFIGS = {
    'vera': VERA_TRAINED,
    'scut': SCUT_TRAINED,
}

DATASET_INFO = {
    'vera': {'dir': VERA_DIR, 'name': 'VERA', 'desc': 'VERA (220 identities, finger vein)'},
    'scut': {'dir': SCUT_DIR, 'name': 'SCUT', 'desc': 'SCUT (1100 identities, palm vein)'},
}


# =====================================================================
# Dataset
# =====================================================================
class CrossDomainDataset(Dataset):
    def __init__(self, data_dirs, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        all_identity_dirs = []
        for d in data_dirs:
            d = Path(d)
            if d.exists():
                for identity_dir in sorted(d.iterdir()):
                    if identity_dir.is_dir():
                        all_identity_dirs.append(identity_dir)

        unique_identities = sorted(set(d.name for d in all_identity_dirs))
        self.class_to_idx = {name: idx for idx, name in enumerate(unique_identities)}

        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        for identity_dir in all_identity_dirs:
            label = self.class_to_idx[identity_dir.name]
            for img_path in sorted(identity_dir.iterdir()):
                if img_path.suffix.lower() in valid_exts:
                    self.samples.append((str(img_path), label))

        print(f"  Loaded {len(self.samples)} images from {len(self.class_to_idx)} identities")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label


class CLAHETransform:
    def __call__(self, image):
        import cv2
        img_np = np.array(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_np)
        return Image.fromarray(enhanced)


# =====================================================================
# Model Loading
# =====================================================================
def load_model(config, feature_dim=1024):
    checkpoint_path = project_root / config['checkpoint']

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"  Loading {config['name']} from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    num_classes = config['num_train_classes']
    model_type = config['model_type']

    if model_type == 'sca_mobilenet':
        from SCA_MobileNet.model import SCAMobileNet
        model = SCAMobileNet(
            embedding_size=feature_dim,
            class_size=num_classes,
            pretrained=False,
            only_embeddings=True,
            use_stn=True, use_ca=True, use_spp=True,
            dropout=0.3
        )

    elif model_type == 'mobilenetv3_base':
        from SCA_MobileNet.model import SCAMobileNet
        model = SCAMobileNet(
            embedding_size=feature_dim,
            class_size=num_classes,
            pretrained=False,
            only_embeddings=True,
            use_stn=False, use_ca=False, use_spp=False,
            dropout=0.3
        )

    elif model_type == 'mpsnet':
        from MPSNet_2022.model_pytorch import MPSNet
        model = MPSNet(
            feature_dim=feature_dim,
            input_channels=1,
            dropout=0.3
        )

    elif model_type == 'eusipco2020':
        eusipco_dir = project_root / 'models' / 'Modified_Densenet161_2021'
        spec = importlib.util.spec_from_file_location(
            "modified_models",
            eusipco_dir / 'models' / 'modified_models.py'
        )
        modified_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modified_models)
        model = modified_models.DenseNet161_Modified(
            embedding_size=feature_dim,
            class_size=num_classes,
            pretrained=False,
            only_embeddings=True,
            l2_normed=True
        )

    elif model_type == 'rsnet':
        from RSNet.model import RSNet
        model = RSNet(
            feature_dim=feature_dim,
            in_channels=3,
            dropout_rate=0.3
        )

    elif model_type == 'fgfnet':
        from FGFNet.model import MobileViT_FFC_ATTN_FFTSA
        model = MobileViT_FFC_ATTN_FFTSA(
            image_size=(256, 256),
            num_classes=num_classes
        )

    elif model_type == 'sca_transformer':
        from train import SCATransformerBackbone
        model = SCATransformerBackbone(
            backbone_name=config['backbone_name'],
            embedding_size=feature_dim,
            dropout=0.3,
            pretrained=False
        )

    elif model_type == 'gscl':
        from GSCL_2024.models.models import ResNets
        model = ResNets(
            backbone=config['backbone_name'],
            head_type='cls_norm',
            num_classes=num_classes
        )

    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()

    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model


# =====================================================================
# Embedding Extraction
# =====================================================================
def extract_embeddings(model, dataloader, model_type):
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Extracting', leave=False):
            images = images.to(DEVICE, non_blocking=True)

            if model_type == 'eusipco2020':
                embeddings = model(images, train=False)
            elif model_type == 'fgfnet':
                embeddings = model.get_embedding(images)
            elif model_type == 'sca_transformer':
                embeddings = model(images)
            else:
                model_out = model(images)
                if isinstance(model_out, tuple):
                    embeddings = model_out[0]
                else:
                    embeddings = model_out

            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_embeddings), np.concatenate(all_labels)


# =====================================================================
# Main
# =====================================================================
def evaluate_single(model_key, source, target, feature_dim=1024, batch_size=64):
    configs = SOURCE_CONFIGS[source]
    if model_key not in configs:
        print(f"  [SKIP] {model_key} — no checkpoint for {source.upper()}-trained model")
        return None

    config = configs[model_key]
    target_info = DATASET_INFO[target]
    source_info = DATASET_INFO[source]

    print("=" * 70)
    print(f"CROSS-DOMAIN: {config['name']} (Train {source_info['name']} -> Test {target_info['name']})")
    print("=" * 70)

    # 1. Load model
    print(f"\n[1/3] Loading model...")
    model = load_model(config, feature_dim=feature_dim)

    # 2. Prepare target dataset (entire dataset: train + test)
    print(f"\n[2/3] Loading entire {target_info['name']} dataset...")
    target_dir = target_info['dir']

    img_size = config['image_size']
    n_channels = config['input_channels']

    transform_list = [CLAHETransform(), transforms.Resize((img_size, img_size))]
    if n_channels == 3:
        transform_list.append(transforms.Grayscale(num_output_channels=3))
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406][:n_channels],
            std=[0.229, 0.224, 0.225][:n_channels]
        )
    ])
    test_transform = transforms.Compose(transform_list)

    dataset = CrossDomainDataset(
        [target_dir / "train", target_dir / "test"],
        transform=test_transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # 3. Extract & Evaluate
    print(f"\n[3/3] Extracting embeddings & computing metrics...")
    embeddings, labels = extract_embeddings(model, dataloader, config['model_type'])

    print(f"  {len(embeddings)} embeddings, dim={embeddings.shape[1]}, "
          f"{len(np.unique(labels))} identities")

    evaluator = BiometricEvaluator()
    results = evaluator.evaluate_verification(embeddings, labels)

    # Print results
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {config['name']} ({source_info['name']} -> {target_info['name']})")
    print(f"{'=' * 70}")
    print(f"  EER:            {results['EER']*100:.4f}%")
    print(f"  AUC:            {results['AUC']:.6f}")
    print(f"  TAR@0.01% FAR:  {results['TAR@0.01%FAR']*100:.3f}%")
    print(f"  TAR@0.1% FAR:   {results['TAR@0.1%FAR']*100:.3f}%")
    print(f"  TAR@1% FAR:     {results['TAR@1%FAR']*100:.3f}%")
    print(f"  D-prime:        {results['d_prime']:.4f}")
    print(f"{'=' * 70}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": f"cross_domain_{source}_to_{target}_{model_key}",
        "timestamp": datetime.now().isoformat(),
        "model": config['name'],
        "train_dataset": source_info['desc'],
        "test_dataset": target_info['desc'],
        "checkpoint": config['checkpoint'],
        "metrics": {
            "eer": float(results['EER']),
            "auc": float(results['AUC']),
            "tar_at_001_far": float(results['TAR@0.01%FAR']),
            "tar_at_01_far": float(results['TAR@0.1%FAR']),
            "tar_at_1_far": float(results['TAR@1%FAR']),
            "d_prime": float(results['d_prime']),
            "genuine_mean": float(results['genuine_mean']),
            "imposter_mean": float(results['imposter_mean']),
            "n_genuine_pairs": int(results['n_genuine_pairs']),
            "n_imposter_pairs": int(results['n_imposter_pairs'])
        }
    }

    output_file = OUTPUT_DIR / f"cross_domain_{source}_to_{target}_{model_key}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_file}")
    return output


def main():
    parser = argparse.ArgumentParser(description='Cross-Domain Evaluation: VERA <-> SCUT')
    parser.add_argument('--direction', type=str, required=True,
                        choices=['vera_to_scut', 'scut_to_vera'],
                        help='Cross-domain direction')
    parser.add_argument('--model', type=str, default=None,
                        help='Model to evaluate (omit for --all)')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all available models')
    parser.add_argument('--feature-dim', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    source, target = args.direction.split('_to_')

    if args.all or args.model is None:
        configs = SOURCE_CONFIGS[source]
        all_results = []
        for model_key in configs:
            try:
                result = evaluate_single(model_key, source, target,
                                         args.feature_dim, args.batch_size)
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"  [ERROR] {model_key}: {e}")
            print()

        # Summary table
        if all_results:
            print("\n" + "=" * 80)
            print(f"SUMMARY: {source.upper()} -> {target.upper()} Cross-Domain")
            print("=" * 80)
            print(f"{'Model':<25} {'EER%':>8} {'TAR@0.01%':>10} {'TAR@0.1%':>10} {'TAR@1%':>10} {'D-prime':>8}")
            print("-" * 80)
            for r in sorted(all_results, key=lambda x: x['metrics']['eer']):
                m = r['metrics']
                print(f"{r['model']:<25} {m['eer']*100:>7.3f}% {m['tar_at_001_far']*100:>9.2f}% "
                      f"{m['tar_at_01_far']*100:>9.2f}% {m['tar_at_1_far']*100:>9.2f}% "
                      f"{m['d_prime']:>8.3f}")
            print("=" * 80)

            summary_file = OUTPUT_DIR / f"summary_{source}_to_{target}.json"
            with open(summary_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Summary saved to {summary_file}")
    else:
        evaluate_single(args.model, source, target,
                         args.feature_dim, args.batch_size)


if __name__ == "__main__":
    main()
