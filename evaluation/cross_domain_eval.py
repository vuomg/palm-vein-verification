"""
Cross-Domain Evaluation Script (All Models)
=============================================
Train on DTS (self-collected palm vein) → Test on TONGJI (palmprint)

Usage:
    python cross_domain_eval.py --model sca_mobilenet
    python cross_domain_eval.py --model mpsnet
    python cross_domain_eval.py --model eusipco2020
    python cross_domain_eval.py --model rsnet
    python cross_domain_eval.py --model fgfnet

This script loads a model trained on the self-collected DTS dataset
and evaluates it directly on the entire TONGJI dataset (1200 identities).
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

# Add parent dir for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'models'))

from biometric.metrics import BiometricEvaluator

# =====================================================================
# Configuration
# =====================================================================
TONGJI_DIR = Path("TONGJI_dataset_openset")
OUTPUT_DIR = Path("results_cross_domain")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoint mapping (DTS-trained models)
MODEL_CONFIGS = {
    'sca_mobilenet': {
        'name': 'SCA-MobileNet',
        'checkpoint': 'results_sca_v2_sca_mobilenet/best_sca_mobilenet_model_eer.pth',
        'input_channels': 3,
        'image_size': 224,
        'model_type': 'sca_mobilenet',
    },
    'mpsnet': {
        'name': 'MPSNet',
        'checkpoint': 'results_mpsnet/best_mpsnet_model_eer.pth',
        'input_channels': 1,
        'image_size': 224,
        'model_type': 'mpsnet',
    },
    'eusipco2020': {
        'name': 'Modified-DenseNet161',
        'checkpoint': 'results_eusipco2020/checkpoints/best_model_seed99.pth',
        'input_channels': 3,
        'image_size': 224,
        'model_type': 'eusipco2020',
    },
    'rsnet': {
        'name': 'RSNet',
        'checkpoint': 'results_rsnet/best_rsnet_model_eer.pth',
        'input_channels': 3,
        'image_size': 224,
        'model_type': 'rsnet',
    },
    'fgfnet': {
        'name': 'FGFNet',
        'checkpoint': 'results_fgfnet/best_fgfnet_model_eer.pth',
        'input_channels': 3,
        'image_size': 256,
        'model_type': 'fgfnet',
    },
    'gscl': {
        'name': 'GSCL',
        'checkpoint': 'models/GSCL-PyTorch/vein_feature_learning/results/palmvein_resnet18/checkpoints/best_model_seed42.pth',
        'input_channels': 3,
        'image_size': 224,
        'model_type': 'gscl',
    },
}


# =====================================================================
# Dataset
# =====================================================================
class CrossDomainDataset(Dataset):
    """Load all images from multiple subdirectories as a single flat dataset."""
    
    def __init__(self, data_dirs, transform=None, grayscale=True):
        self.transform = transform
        self.grayscale = grayscale
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
        
        self.data_dir = data_dirs[0]
        print(f"  Loaded {len(self.samples)} images from {len(self.class_to_idx)} identities")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label


# =====================================================================
# CLAHE Transform
# =====================================================================
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
def load_model(model_key, num_classes=1084, feature_dim=1024):
    """Load a model and its DTS-trained checkpoint."""
    config = MODEL_CONFIGS[model_key]
    checkpoint_path = Path(config['checkpoint'])
    
    print(f"  Loading {config['name']} from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Handle checkpoint format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    if model_key == 'sca_mobilenet':
        from SCA_MobileNet.model import SCAMobileNet
        model = SCAMobileNet(
            embedding_size=feature_dim,
            class_size=num_classes,
            pretrained=False,
            only_embeddings=True,
            use_stn=True,
            use_ca=True,
            use_spp=True,
            dropout=0.3
        )
        
    elif model_key == 'mpsnet':
        from MPSNet_2022.model_pytorch import MPSNet
        model = MPSNet(
            feature_dim=feature_dim,
            input_channels=1,
            dropout=0.3
        )
        
    elif model_key == 'eusipco2020':
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
        
    elif model_key == 'rsnet':
        from RSNet.model import RSNet
        model = RSNet(
            feature_dim=feature_dim,
            in_channels=3,
            dropout_rate=0.3
        )
        
    elif model_key == 'fgfnet':
        from FGFNet.model import MobileViT_FFC_ATTN_FFTSA
        model = MobileViT_FFC_ATTN_FFTSA(
            image_size=(256, 256),
            num_classes=num_classes
        )
    
    elif model_key == 'gscl':
        from GSCL_2024.models.models import ResNets
        model = ResNets(
            backbone='resnet18',
            head_type='cls_norm',
            num_classes=num_classes
        )
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()
    
    print(f"  Model loaded successfully ({sum(p.numel() for p in model.parameters()):,} params)")
    return model, config


# =====================================================================
# Embedding Extraction
# =====================================================================
def extract_embeddings(model, dataloader, model_type):
    """Extract L2-normalized embeddings."""
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Extracting', leave=False):
            images = images.to(DEVICE, non_blocking=True)
            
            if model_type == 'eusipco2020':
                embeddings = model(images, train=False)
            elif model_type == 'fgfnet':
                embeddings = model.get_embedding(images)
            elif model_type == 'gscl':
                embeddings, _ = model(images)
            else:  # rsnet, mpsnet, sca_mobilenet
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
def main():
    parser = argparse.ArgumentParser(description='Cross-Domain Evaluation: DTS → TONGJI')
    parser.add_argument('--model', type=str, required=True,
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Model to evaluate')
    parser.add_argument('--feature-dim', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    
    config = MODEL_CONFIGS[args.model]
    
    print("=" * 70)
    print(f"CROSS-DOMAIN: {config['name']} (Train DTS → Test TONGJI)")
    print("=" * 70)
    
    # 1. Load Model
    print(f"\n[1/3] Loading model...")
    model, config = load_model(args.model, feature_dim=args.feature_dim)
    
    # 2. Prepare TONGJI Dataset
    print(f"\n[2/3] Loading entire TONGJI dataset...")
    
    img_size = config['image_size']
    n_channels = config['input_channels']
    
    transform_list = [CLAHETransform(), transforms.Resize((img_size, img_size))]
    
    if n_channels == 3:
        transform_list.append(transforms.Grayscale(num_output_channels=3))
    else:
        # 1 channel
        pass
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406][:n_channels],
            std=[0.229, 0.224, 0.225][:n_channels]
        )
    ])
    
    test_transform = transforms.Compose(transform_list)
    
    dataset = CrossDomainDataset(
        [TONGJI_DIR / "train", TONGJI_DIR / "test"],
        transform=test_transform
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
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
    print(f"RESULTS: {config['name']} (DTS → TONGJI)")
    print(f"{'=' * 70}")
    print(f"  EER:            {results['EER']*100:.4f}%")
    print(f"  AUC:            {results['AUC']:.6f}")
    print(f"  TAR@0.01% FAR:  {results['TAR@0.01%FAR']*100:.3f}%")
    print(f"  TAR@0.1% FAR:   {results['TAR@0.1%FAR']*100:.3f}%")
    print(f"  TAR@1% FAR:     {results['TAR@1%FAR']*100:.3f}%")
    print(f"  D-prime:        {results['d_prime']:.4f}")
    print(f"{'=' * 70}")
    
    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    output = {
        "experiment": f"cross_domain_{args.model}",
        "timestamp": datetime.now().isoformat(),
        "model": config['name'],
        "train_dataset": "DTS (self-collected, 1084 train identities)",
        "test_dataset": "TONGJI (1200 identities, all used)",
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
    
    output_file = OUTPUT_DIR / f"cross_domain_{args.model}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
