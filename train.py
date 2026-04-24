"""
Multi-Model Training Script for Palm Vein Authentication
Supports: RSNet-RLEB, EUSIPCO 2020, MPSNet, GSCL, MobileNetV3-Small + SPP, VeinKAN

Usage:
    python train.py --model rsnet --dataset path/to/dataset
    python train.py --model eusipco2020 --dataset path/to/dataset
    python train.py --model mpsnet --dataset path/to/dataset
    python train.py --model gscl --dataset path/to/dataset --gscl-backbone resnet18
    python train.py --model mobilenetv3small --dataset path/to/dataset
    python train.py --model veinkan --dataset path/to/dataset
"""

import os
import sys
import time
import argparse
import logging
import random
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

# Add current project directory for local biometric package (priority)
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / 'models'))

# Import from biometric package
from biometric.data_augmentation import BiometricCompose, CarefulRotation, CarefulTranslation
from biometric.data_augmentation import GaussianNoise, ContrastAdjustment, BrightnessAdjustment, RandomScale
from biometric.data_augmentation import RandomPerspectiveTransformation, RandomGammaAdjustment
from biometric.early_stopping import EarlyStopping, TrainingMonitor
from biometric.metrics import BiometricEvaluator

# Import RSNet components
from RSNet.model import RSNet
from RSNet.losses import AdaFaceLoss, RSNetJointLoss, get_database_config

# --- SCA-MobileNet (Spatial-Coordinate Attention MobileNet) ---
from SCA_MobileNet.model import SCAMobileNet
from SCA_MobileNet.losses import FusionLoss, BalancedBatchSampler
from SCA_MobileNet.losses_adacos import AdaCos as AdaCosLoss

# --- FGFNet (MobileViT + FFC + FFT) ---
from FGFNet.model import MobileViT_FFC_ATTN_FFTSA
from FGFNet.loss import FGFNetLoss
from biometric.visualization import generate_training_charts

# Import dataset and transforms
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

try:
    import timm
except Exception:
    timm = None


class SCATransformerBackbone(nn.Module):
    """Alternative SCA backbone using timm ImageNet-pretrained models."""

    BACKBONE_MAP = {
        'mobilevit_s': 'mobilevit_s.cvnets_in1k',
        'deit_tiny': 'deit_tiny_patch16_224.fb_in1k',
        'swin_tiny': 'swin_tiny_patch4_window7_224.ms_in1k',
        'efficientnet_b0': 'efficientnet_b0.ra_in1k',
    }

    def __init__(self, backbone_name='mobilevit_s', embedding_size=1024, dropout=0.3, pretrained=True):
        super().__init__()
        if timm is None:
            raise ImportError(
                "timm is required for --sca-backbone options. "
                "Install with: pip install timm"
            )
        if backbone_name not in self.BACKBONE_MAP:
            raise ValueError(
                f"Unsupported sca-backbone: {backbone_name}. "
                f"Supported: {list(self.BACKBONE_MAP.keys())}"
            )

        timm_model_name = self.BACKBONE_MAP[backbone_name]
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            timm_model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        backbone_dim = getattr(self.backbone, 'num_features', None)
        if backbone_dim is None:
            raise RuntimeError(f"Could not infer num_features for {timm_model_name}")

        self.embedder = nn.Sequential(
            nn.BatchNorm1d(backbone_dim),
            nn.Dropout(p=dropout),
            nn.Linear(backbone_dim, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        feats = self.backbone(x)
        embeddings = self.embedder(feats)
        return embeddings

    def freeze_backbone(self):
        """Freeze backbone, only train embedder head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.embedder.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def get_layer_wise_param_groups(self, base_lr, decay=0.75):
        """
        Build parameter groups with layer-wise LR decay for transformer backbones.
        Deeper (later) layers get higher LR; earlier layers get lower LR.
        The embedder head always gets the full base_lr.

        Args:
            base_lr: LR for the embedder head and last backbone layer group
            decay: multiplicative decay applied per layer group (e.g. 0.75)

        Returns:
            list of param group dicts for the optimizer
        """
        backbone = self.backbone
        backbone_name = self.backbone_name

        if backbone_name == 'swin_tiny':
            layer_groups = []
            # patch_embed (earliest)
            layer_groups.append(
                [p for p in backbone.patch_embed.parameters() if p.requires_grad]
            )
            # layers / stages (4 stages for Swin-Tiny)
            for stage in backbone.layers:
                layer_groups.append(
                    [p for p in stage.parameters() if p.requires_grad]
                )
            # norm at the end of backbone
            if hasattr(backbone, 'norm'):
                layer_groups.append(
                    [p for p in backbone.norm.parameters() if p.requires_grad]
                )
        elif backbone_name == 'deit_tiny':
            layer_groups = []
            layer_groups.append(
                [p for p in backbone.patch_embed.parameters() if p.requires_grad]
            )
            if hasattr(backbone, 'blocks'):
                n_blocks = len(backbone.blocks)
                group_size = max(1, n_blocks // 4)
                for i in range(0, n_blocks, group_size):
                    params = []
                    for block in backbone.blocks[i:i + group_size]:
                        params.extend(
                            [p for p in block.parameters() if p.requires_grad]
                        )
                    layer_groups.append(params)
            if hasattr(backbone, 'norm'):
                layer_groups.append(
                    [p for p in backbone.norm.parameters() if p.requires_grad]
                )
        else:
            # mobilevit_s or generic: split into stem vs stages
            layer_groups = []
            seen = set()
            for name, param in backbone.named_parameters():
                if not param.requires_grad:
                    continue
                if 'stem' in name or 'conv_stem' in name or 'patch_embed' in name:
                    layer_groups.append([param])
                    seen.add(id(param))
            remaining = [p for p in backbone.parameters()
                         if p.requires_grad and id(p) not in seen]
            if remaining:
                mid = len(remaining) // 2
                layer_groups.append(remaining[:mid])
                layer_groups.append(remaining[mid:])

        # Filter out empty groups
        layer_groups = [g for g in layer_groups if len(g) > 0]

        # Assign decayed LRs: earliest layer group gets the smallest LR
        n_groups = len(layer_groups)
        param_groups = []
        for i, params in enumerate(layer_groups):
            lr = base_lr * (decay ** (n_groups - 1 - i))
            param_groups.append({'params': params, 'lr': lr})

        # Embedder head always gets full base_lr
        head_params = [p for p in self.embedder.parameters() if p.requires_grad]
        if head_params:
            param_groups.append({'params': head_params, 'lr': base_lr})

        return param_groups


# --- SUPPORT FUNCTION FOR FREEZE/UNFREEZE WARM-UP STRATEGY ---
def set_training_mode(model, mode='warmup'):
    """
    Controls layer freezing for Warm-up/Fine-tune strategy.
    
    Args:
        model: SCAMobileNet model
        mode: 'warmup' to freeze backbone, 'finetune' to unfreeze all layers
    """
    if mode == 'warmup':
        # 1. Freeze all features first
        for param in model.features.parameters():
            param.requires_grad = False
        
        # 2. Unfreeze STN (if applicable)
        if hasattr(model, 'use_stn') and model.use_stn:
            for param in model.stn.parameters():
                param.requires_grad = True
                
        # 3. Unfreeze Coordinate Attention (Index 9 in model.features)
        # Note: Carefully check index in model.features
        if len(model.features) > 9:
            for param in model.features[9].parameters():
                param.requires_grad = True
            
        # 4. Unfreeze Head (embedder)
        for param in model.embedder.parameters():
            param.requires_grad = True
        
        # 5. Unfreeze SPP components (if applicable)
        if hasattr(model, 'spp_bn'):
            for param in model.spp_bn.parameters():
                param.requires_grad = True
            
    elif mode == 'finetune':
        for param in model.parameters():
            param.requires_grad = True


class PalmVeinDataset(Dataset):
    """Custom dataset for palm vein recognition."""
    
    def __init__(self, data_dir, transform=None, max_samples_per_class=None, valid_classes=None, image_mode='L'):
        """
        Args:
            data_dir: Path to dataset directory
            transform: Transforms to apply to images
            max_samples_per_class: Limit samples per identity (for debugging)
            valid_classes: Dict of valid class names to indices (for filtering test set)
            image_mode: 'L' for grayscale (1-channel), 'RGB' for 3-channel
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
        self.valid_classes = valid_classes
        self.image_mode = image_mode
        
        # Load dataset
        self.samples = []
        self.class_to_idx = {}
        self._load_dataset()
        
        print(f"Loaded {len(self.samples)} samples from {len(self.class_to_idx)} classes")
    
    def _load_dataset(self):
        """Load dataset from directory structure."""
        class_idx = 0
        
        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            if self.valid_classes is not None:
                if class_name not in self.valid_classes:
                    continue
                self.class_to_idx[class_name] = self.valid_classes[class_name]
            else:
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = class_idx
                    class_idx += 1
            
            image_files = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.bmp'))
            
            if self.max_samples_per_class:
                image_files = image_files[:self.max_samples_per_class]
            
            for img_path in image_files:
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert(self.image_mode)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_embeddings(model, dataloader, device, model_type='rsnet'):
    """
    Extract embeddings from model for evaluation.
    
    Args:
        model: Model (RSNet or EUSIPCO 2020)
        dataloader: DataLoader for dataset
        device: Device to run on
        model_type: 'rsnet' or 'eusipco2020'
        
    Returns:
        embeddings: numpy array (N, D)
        labels: numpy array (N,)
        image_paths: numpy array (N,) - relative paths
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    all_image_paths = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc='Extracting embeddings', leave=False)):
            images = images.to(device, non_blocking=True)
            
            # Get embeddings based on model type
            if model_type == 'eusipco2020':
                embeddings = model(images, train=False)
            elif model_type == 'gscl':
                # GSCL returns (embeddings, logits) - we only need embeddings
                embeddings, _ = model(images)
            elif model_type == 'veinkan':
                # VeinKAN: use get_embedding() to extract backbone features (2048-d)
                embeddings = model.get_embedding(images)
            elif model_type == 'fgfnet':
                # FGFNet: use get_embedding() to extract features before FC
                embeddings = model.get_embedding(images)
            else:  # rsnet, mpsnet, sca_mobilenet, etc.
                model_out = model(images)
                # Handle tuple output (embeddings, theta) from STN-enabled models
                if isinstance(model_out, tuple):
                    embeddings = model_out[0]
                else:
                    embeddings = model_out
            
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
            
            # Extract image paths
            batch_size = len(labels)
            start_idx = batch_idx * dataloader.batch_size
            for i in range(batch_size):
                idx = start_idx + i
                if idx < len(dataloader.dataset.samples):
                    img_path, _ = dataloader.dataset.samples[idx]
                    rel_path = str(Path(img_path).relative_to(dataloader.dataset.data_dir))
                    all_image_paths.append(rel_path)
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    image_paths = np.array(all_image_paths)
    
    return embeddings, labels, image_paths


def evaluate_verification_metrics(model, test_loader, device, epoch, balanced_sampling=True, model_type='rsnet'):
    """
    Evaluate verification metrics (Paper: Section IV-A).
    Metrics: EER, AUC, TAR@FAR, DIR@FPIR per paper protocol.
    
    Args:
        model: Model (RSNet or EUSIPCO 2020)
        test_loader: Test set data loader
        device: Device
        epoch: Current epoch
        balanced_sampling: Use 1:1 genuine-imposter ratio (paper strategy)
        model_type: 'rsnet' or 'eusipco2020'
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n{'='*70}")
    print(f"VERIFICATION EVALUATION - Epoch {epoch}")
    print(f"{'='*70}")
    
    # Extract embeddings from test set
    embeddings, labels, image_paths = extract_embeddings(model, test_loader, device, model_type)
    
    print(f"Extracted {len(embeddings)} embeddings from {len(np.unique(labels))} users")
    
    # Evaluate using BiometricEvaluator with paper strategy
    evaluator = BiometricEvaluator()
    
    # Paper uses balanced 1:1 genuine-imposter sampling
    results = evaluator.evaluate_verification(
        embeddings, labels
    )
    
    # Print key metrics (Paper: Table II, III)
    print(f"\nPAPER METRICS (Section IV-A):")
    print(f"  EER (Equal Error Rate):           {results['EER']*100:.4f}%")
    print(f"  AUC (Area Under Curve):           {results['AUC']:.6f}")
    print(f"\nOPERATING POINTS (TAR@FAR):")
    print(f"  TAR @ FAR=0.01%:                  {results['TAR@0.01%FAR']*100:.2f}%")
    print(f"  TAR @ FAR=0.1%:                   {results['TAR@0.1%FAR']*100:.2f}%")
    print(f"  TAR @ FAR=1%:                     {results['TAR@1%FAR']*100:.2f}%")
    print(f"\nSCORE STATISTICS:")
    print(f"  Genuine mean ± std:               {results['genuine_mean']:.4f} ± {results['genuine_std']:.4f}")
    print(f"  Imposter mean ± std:              {results['imposter_mean']:.4f} ± {results['imposter_std']:.4f}")
    print(f"  D-prime (separability):           {results['d_prime']:.4f}")
    
    # User rejection analysis
    thresholds = {
        'far_001': results['threshold_001_far'],
        'far_01': results['threshold_01_far'],
        'far_1': results['threshold_1_far']
    }
    rejection_analysis = evaluator.analyze_user_rejections(embeddings, labels, thresholds, image_paths)
    print(f"{'='*70}\n")
    
    # Return metrics for logging
    return {
        'eer': results['EER'],
        'auc': results['AUC'],
        'fnir_at_001_fpir': results['FNIR@0.01%FPIR'],
        'fnir_at_01_fpir': results['FNIR@0.1%FPIR'],
        'fnir_at_1_fpir': results['FNIR@1%FPIR'],
        'tar_at_001_far': results['TAR@0.01%FAR'],
        'tar_at_01_far': results['TAR@0.1%FAR'],
        'tar_at_1_far': results['TAR@1%FAR'],
        'd_prime': results['d_prime'],
        'genuine_mean': results['genuine_mean'],
        'imposter_mean': results['imposter_mean'],
        'n_genuine_pairs': results['n_genuine_pairs'],
        'n_imposter_pairs': results['n_imposter_pairs'],
        'rejection_analysis': rejection_analysis
    }


def save_metrics_to_json(metrics_dict, output_dir, epoch):
    """Save training metrics to JSON file for analysis."""
    output_path = Path(output_dir)
    metrics_file = output_path / 'training_metrics.json'
    
    # Convert numpy types to native Python types
    def convert_to_native(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(i) for i in obj]
        return obj
    
    # Extract rejection analysis (if present) and save separately
    rejection_analysis = metrics_dict.pop('rejection_analysis', None)
    
    # Load existing metrics if file exists
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            # Corrupted file, backup and restart
            print(f"  >> Warning: Corrupted metrics file, creating backup and restarting")
            backup_file = output_path / f'training_metrics_backup_{epoch}.json'
            metrics_file.rename(backup_file)
            all_metrics = {'epochs': []}
    else:
        all_metrics = {'epochs': []}
    
    # Add current epoch metrics (convert numpy types)
    epoch_data = {'epoch': epoch, **convert_to_native(metrics_dict)}
    all_metrics['epochs'].append(epoch_data)
    
    # Save updated metrics
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"  >> Metrics saved to: {metrics_file}")
    
    # Save rejection analysis to separate file
    if rejection_analysis is not None:
        rejection_dir = output_path / 'rejection_analysis'
        rejection_dir.mkdir(exist_ok=True)
        rejection_file = rejection_dir / f'epoch_{epoch}_rejections.json'
        
        rejection_data = {
            'epoch': epoch,
            'analysis': convert_to_native(rejection_analysis)
        }
        
        with open(rejection_file, 'w') as f:
            json.dump(rejection_data, f, indent=2)
        
        print(f"  >> Rejection analysis saved to: {rejection_file}")


def train_epoch_rsnet(model, train_loader, joint_loss, optimizer, scaler, device, epoch):
    """Train one epoch with RSNet joint loss."""
    model.train()
    running_loss_total = 0.0
    running_loss_local = 0.0
    running_loss_global = 0.0
    running_loss_diff = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch:3d}')
    
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            # Get dual-branch embeddings
            local_emb, global_emb = model(images)
            
            # Compute joint loss
            total_loss, loss_local, loss_global, loss_diff = joint_loss(
                local_emb, global_emb, labels
            )
        
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        
        # Gradient clipping (handles large gradients automatically)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        # Check for NaN/Inf and skip batch if detected
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            optimizer.zero_grad()
            continue
        
        scaler.step(optimizer)
        scaler.update()
        
        running_loss_total += total_loss.item()
        running_loss_local += loss_local.item()
        running_loss_global += loss_global.item()
        running_loss_diff += loss_diff.item()
        num_batches += 1
        
        progress_bar.set_postfix({
            'Total': f'{total_loss.item():.4f}',
            'Local': f'{loss_local.item():.4f}',
            'Global': f'{loss_global.item():.4f}',
            'Diff': f'{loss_diff.item():.6f}',
            'D/L_ratio': f'{loss_diff.item()/loss_local.item():.3f}'  # Monitor diff loss magnitude
        })
    
    return {
        'total': running_loss_total / num_batches,
        'local': running_loss_local / num_batches,
        'global': running_loss_global / num_batches,
        'diff': running_loss_diff / num_batches
    }


def validate_rsnet(model, val_loader, joint_loss, device):
    """Validate RSNet model with both branches."""
    model.eval()
    running_loss_total = 0.0
    running_loss_local = 0.0
    running_loss_global = 0.0
    running_loss_diff = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation', leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Temporarily set to training mode to get both branches
            model.train()
            local_emb, global_emb = model(images)
            model.eval()
            
            total_loss, loss_local, loss_global, loss_diff = joint_loss(
                local_emb, global_emb, labels
            )
            
            running_loss_total += total_loss.item()
            running_loss_local += loss_local.item()
            running_loss_global += loss_global.item()
            running_loss_diff += loss_diff.item()
            num_batches += 1
    
    if num_batches == 0:
        print("⚠️  WARNING: Validation set is empty!")
        return {
            'val_loss': float('inf'),
            'val_loss_local': float('inf'),
            'val_loss_global': float('inf'),
            'val_loss_diff': float('inf')
        }
    
    return {
        'val_loss': running_loss_total / num_batches,
        'val_loss_local': running_loss_local / num_batches,
        'val_loss_global': running_loss_global / num_batches,
        'val_loss_diff': running_loss_diff / num_batches
    }


def train_epoch_single_branch(model, train_loader, margin_loss, ce_loss, optimizer, scaler, device, epoch, model_type='eusipco2020', lambda_geo=0.1):
    """Train one epoch for single-branch models (EUSIPCO types, MPSNet, MobileNetV3+AdaCos)."""
    model.train()
    running_loss = 0.0
    running_geo_loss = 0.0
    num_batches = 0
    
    # Check if model has geometric regularization (STN with geo_reg_loss)
    has_geo_reg = hasattr(model, 'use_stn') and model.use_stn and hasattr(model, 'geo_reg_loss')
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch:3d} ({model_type})')
    
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            # Get embeddings from model
            theta = None
            if model_type == 'eusipco2020':
                model_out = model(images, train=True)
            else:
                model_out = model(images)
            
            # Handle tuple output (embeddings, theta) from STN-enabled models
            if isinstance(model_out, tuple):
                embeddings, theta = model_out
            else:
                embeddings = model_out
            
            # Compute ArcFace logits and cross-entropy loss
            logits = margin_loss(embeddings, labels)
            total_loss = ce_loss(logits, labels)
            
            # Add geometric regularization loss if STN is active
            geo_loss = torch.tensor(0.0, device=device)
            if has_geo_reg and theta is not None:
                geo_loss = model.geo_reg_loss(theta)
                total_loss = total_loss + lambda_geo * geo_loss
        
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        # Check for NaN/Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            optimizer.zero_grad()
            continue
        
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += total_loss.item()
        running_geo_loss += geo_loss.item()
        num_batches += 1
        
        postfix = {'Loss': f'{total_loss.item():.4f}'}
        if has_geo_reg:
            postfix['Geo'] = f'{geo_loss.item():.4f}'
        progress_bar.set_postfix(postfix)
    
    result = {'total': running_loss / max(num_batches, 1)}
    if has_geo_reg:
        result['geo_loss'] = running_geo_loss / max(num_batches, 1)
    return result


def validate_single_branch(model, val_loader, margin_loss, ce_loss, device, model_type='eusipco2020'):
    """Validate single-branch model (EUSIPCO types, MPSNet)."""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation', leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Get embeddings
            if model_type == 'eusipco2020':
                model.train()
                model_out = model(images, train=True)
                model.eval()
            else:
                model_out = model(images)
            
            # Handle tuple output (embeddings, theta) from STN-enabled models
            if isinstance(model_out, tuple):
                embeddings, _ = model_out
            else:
                embeddings = model_out
            
            # Compute loss
            logits = margin_loss(embeddings, labels)
            total_loss = ce_loss(logits, labels)
            
            running_loss += total_loss.item()
            num_batches += 1
    
    if num_batches == 0:
        return {'val_loss': float('inf')}
    
    return {'val_loss': running_loss / num_batches}


def train_epoch_gscl(model, train_loader, fusion_loss, optimizer, scaler, device, epoch):
    """Train one epoch for GSCL model with FusionLoss (CosFace + TripletLoss)."""
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch:3d} (GSCL)')
    
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            # GSCL model returns (embeddings, logits)
            embeddings, logits = model(images)
            
            # FusionLoss expects (embeddings, logits) as first argument
            total_loss = fusion_loss((embeddings, logits), labels)
        
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        # Check for NaN/Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            optimizer.zero_grad()
            continue
        
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += total_loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'Loss': f'{total_loss.item():.4f}'})
    
    return {'total': running_loss / max(num_batches, 1)}


def validate_gscl(model, val_loader, fusion_loss, device):
    """Validate GSCL model with FusionLoss."""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation', leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Get embeddings and logits
            embeddings, logits = model(images)
            
            # Compute FusionLoss
            total_loss = fusion_loss((embeddings, logits), labels)
            
            running_loss += total_loss.item()
            num_batches += 1
    
    if num_batches == 0:
        return {'val_loss': float('inf')}
    
    return {'val_loss': running_loss / num_batches}


def train_epoch_fusion_v2(model, train_loader, criterion, optimizer, scaler, device, epoch, lambda_geo=0.1):
    """
    Training function for MobileNetV3 + FusionLoss.
    FusionLoss takes (embeddings, labels) instead of (embeddings, logits).
    Supports geometric regularization loss from STN.
    """
    model.train()
    running_loss = 0.0
    running_geo_loss = 0.0
    num_batches = 0
    
    # Check if model has geometric regularization (STN with geo_reg_loss)
    has_geo_reg = hasattr(model, 'use_stn') and model.use_stn and hasattr(model, 'geo_reg_loss')
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch:3d} (Fusion)')
    
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            # 1. Forward Model -> Get Embeddings
            model_out = model(images)
            
            # Handle tuple output (embeddings, theta) from STN-enabled models
            theta = None
            if isinstance(model_out, tuple):
                embeddings, theta = model_out
            else:
                embeddings = model_out
            
            # 2. Compute Fusion Loss (CosFace + Triplet)
            total_loss = criterion(embeddings, labels)
            
            # 3. Add geometric regularization loss if STN is active
            geo_loss = torch.tensor(0.0, device=device)
            if has_geo_reg and theta is not None:
                geo_loss = model.geo_reg_loss(theta)
                total_loss = total_loss + lambda_geo * geo_loss
        
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        # Check for NaN/Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            optimizer.zero_grad()
            continue
        
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += total_loss.item()
        running_geo_loss += geo_loss.item()
        num_batches += 1
        
        postfix = {'Loss': f'{total_loss.item():.4f}'}
        if has_geo_reg:
            postfix['Geo'] = f'{geo_loss.item():.4f}'
        progress_bar.set_postfix(postfix)
    
    result = {'total': running_loss / max(num_batches, 1)}
    if has_geo_reg:
        result['geo_loss'] = running_geo_loss / max(num_batches, 1)
    return result


def validate_fusion_v2(model, val_loader, criterion, device):
    """
    Validate MobileNetV3 + FusionLoss model.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation', leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Get embeddings
            model_out = model(images)
            
            # Handle tuple output (embeddings, theta) from STN-enabled models
            if isinstance(model_out, tuple):
                embeddings, _ = model_out
            else:
                embeddings = model_out
            
            # Compute FusionLoss
            total_loss = criterion(embeddings, labels)
            
            running_loss += total_loss.item()
            num_batches += 1
    
    if num_batches == 0:
        return {'val_loss': float('inf')}
    
    return {'val_loss': running_loss / num_batches}


def train_epoch_fgfnet(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """Train one epoch for FGFNet (Multi-output: logits, feat1, feat2)."""
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_cont = 0.0
    correct = 0
    total = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch:3d} (FGFNet)')
    
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            # FGFNet outputs tuple: (logits, feats1_list, feats2_list)
            outputs = model(images)
            logits = outputs[0]
            
            # Compute combined loss (CE + Contrastive)
            total_loss = criterion(outputs, labels)
            
            # If criterion returns tuple (total, ce, cont)
            if isinstance(total_loss, tuple):
                loss_val, ce_val, cont_val = total_loss
            else:
                loss_val = total_loss
                ce_val = total_loss
                cont_val = torch.tensor(0.0)
        
        scaler.scale(loss_val).backward()
        scaler.unscale_(optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        # Check for NaN/Inf
        if torch.isnan(loss_val) or torch.isinf(loss_val):
            optimizer.zero_grad()
            continue
        
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss_val.item()
        running_ce += ce_val.item()
        running_cont += cont_val.item()
        
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        num_batches += 1
        
        acc = 100. * correct / total
        progress_bar.set_postfix({
            'Loss': f'{loss_val.item():.4f}', 
            'CE': f'{ce_val.item():.4f}',
            'Cont': f'{cont_val.item():.4f}',
            'Acc': f'{acc:.2f}%'
        })
    
    return {
        'total': running_loss / max(num_batches, 1),
        'ce': running_ce / max(num_batches, 1),
        'cont': running_cont / max(num_batches, 1),
        'acc': 100. * correct / max(total, 1)
    }


def validate_fgfnet(model, val_loader, criterion, device):
    """Validate FGFNet model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation', leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward
            outputs = model(images)
            logits = outputs[0]
            
            # Loss
            loss_val = criterion(outputs, labels)
            if isinstance(loss_val, tuple):
                loss_val = loss_val[0]
            
            running_loss += loss_val.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            num_batches += 1
    
    if num_batches == 0:
        return {'val_loss': float('inf'), 'val_acc': 0.0}
    
    return {
        'val_loss': running_loss / num_batches,
        'val_acc': 100. * correct / max(total, 1)
    }


def main():
    parser = argparse.ArgumentParser(description='Multi-Model Training for Palm Vein Authentication')
    
    # Model selection
    parser.add_argument('--model', type=str, default='rsnet',
                       choices=['rsnet', 'eusipco2020', 'mpsnet', 'gscl', 'sca_mobilenet', 'mobilenetv3_uib', 'fgfnet'],
                       help='Model to train: rsnet (default), eusipco2020, mpsnet, gscl, sca_mobilenet, mobilenetv3_uib, or fgfnet')
    
    # EUSIPCO 2020 specific
    parser.add_argument('--eusipco-backbone', type=str, default='densenet161',
                       choices=['densenet161', 'resnext101', 'mnasnet'],
                       help='Backbone for EUSIPCO 2020 model')
    
    # GSCL specific arguments
    parser.add_argument('--gscl-backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='Backbone for GSCL model (default: resnet18)')
    parser.add_argument('--gscl-triplet-margin', type=float, default=0.2,
                       help='Margin for triplet loss in GSCL (default: 0.2)')
    parser.add_argument('--gscl-cosface-s', type=float, default=30.0,
                       help='Scale for CosFace loss in GSCL (default: 30.0)')
    parser.add_argument('--gscl-cosface-m', type=float, default=0.2,
                       help='Margin for CosFace loss in GSCL (default: 0.2)')
    parser.add_argument('--gscl-w-cls', type=float, default=1.0,
                       help='Weight for classification loss in GSCL (default: 1.0)')
    parser.add_argument('--gscl-w-metric', type=float, default=4.0,
                       help='Weight for metric (triplet) loss in GSCL (default: 4.0)')
    
    # SCA-MobileNet specific arguments
    parser.add_argument('--loss-type', type=str, default='adacos_only',
                       choices=['cosface', 'adacos', 'adacos_only'],
                       help='Loss type for sca_mobilenet: cosface, adacos (FusionLoss with AdaCos), or adacos_only (pure AdaCos without Triplet, default)')
    parser.add_argument('--sca-backbone', type=str, default='mobilenetv3',
                       choices=['mobilenetv3', 'mobilevit_s', 'deit_tiny', 'swin_tiny', 'efficientnet_b0'],
                       help='Backbone for sca_mobilenet: mobilenetv3 (default), mobilevit_s, deit_tiny, swin_tiny, efficientnet_b0')
    
    # Ablation study flags (for sca_mobilenet)
    parser.add_argument('--no-stn', action='store_true', default=False,
                       help='Disable STN module (ablation study)')
    parser.add_argument('--no-ca', action='store_true', default=False,
                       help='Disable Coordinate Attention module (ablation study)')
    parser.add_argument('--no-spp', action='store_true', default=False,
                       help='Disable SPP, use GAP instead (ablation study)')
    parser.add_argument('--no-bottleneck', action='store_true', default=False,
                       help='Disable channel bottleneck (576->128), produces ~12M param model')
    parser.add_argument('--ca-reduction', type=int, default=8,
                       help='CoordAttention reduction ratio (default: 8, old 12M used 32)')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset directory (with train/test subdirectories)')
    parser.add_argument('--database', type=str, default='default',
                       choices=['CASIA_850', 'CASIA_940', 'VERA', 'TJ_PV', 
                               'PLUS_PV850', 'PLUS_PV950', 'SCUT', 'default'],
                       help='Database name for specific configuration')
    
    # Model arguments (paper defaults)
    parser.add_argument('--feature-dim', type=int, default=1024,
                       help='Feature dimension (paper: 1024)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--d', type=int, default=2,
                       help='Number of subsets in MAB (paper: 2)')
    parser.add_argument('--n-groups', type=int, default=None,
                       help='Number of groups for channel shuffle (database-specific)')
    
    # Training arguments (paper defaults)
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (paper: 16)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (paper: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate (paper: 0.001)')
    parser.add_argument('--min-lr', type=float, default=0.0001,
                       help='Minimum learning rate for scheduler (paper: 0.0001)')
    parser.add_argument('--T-max', type=int, default=5,
                       help='Cosine annealing T_max (paper: 5)')
    
    # Evaluation strategy (paper: evaluate every 5 epochs, select best EER)
    parser.add_argument('--eval-frequency', type=int, default=5,
                       help='Evaluate on test set every N epochs (paper: 5)')
    parser.add_argument('--use-early-stopping', action='store_true',
                       help='Enable early stopping (NOT used in paper)')
    parser.add_argument('--early-stop-patience', type=int, default=20,
                       help='Early stopping patience if enabled (default: 20)')
    parser.add_argument('--save-best-by', type=str, default='val_loss',
                       choices=['val_loss', 'val_eer'],
                       help='Metric to use for saving best model (default: val_loss)')
    parser.add_argument('--no-checkpoint', action='store_true', default=False,
                       help='Disable periodic epoch checkpoint saves (keeps only best_eer model)')
    
    # Loss arguments (paper defaults - AdaFace)
    parser.add_argument('--adaface-scale', type=float, default=50.0,
                       help='AdaFace scale (s) (paper: 50)')
    parser.add_argument('--adaface-margin', type=float, default=0.55,
                       help='AdaFace margin (m) (paper: 0.55)')
    parser.add_argument('--adaface-h', type=float, default=0.29,
                       help='AdaFace h parameter (paper: 0.29)')
    parser.add_argument('--lambda1', type=float, default=None,
                       help='Weight for global branch loss (database-specific)')
    parser.add_argument('--lambda2', type=float, default=None,
                       help='Weight for difference loss (database-specific)')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Base output directory (model name will be appended)')
    
    args = parser.parse_args()
    
    set_random_seeds(args.seed)
    
    # Update output directory with model name
    base_output_dir = args.output_dir
    vit_name_map = {
        'mobilevit_s': 'MobileViT-S',
        'deit_tiny': 'DeiT-Tiny',
        'swin_tiny': 'Swin-Tiny',
        'efficientnet_b0': 'EfficientNet-B0',
    }
    run_name = args.model
    if args.model == 'sca_mobilenet' and hasattr(args, 'sca_backbone') and args.sca_backbone in vit_name_map:
        run_name = vit_name_map[args.sca_backbone]
    args.output_dir = f"{base_output_dir}_{run_name}"
    
    # Get database-specific configuration (for RSNet)
    db_config = get_database_config(args.database)
    
    # Override with database-specific values if not provided
    if args.lambda1 is None:
        args.lambda1 = db_config['lambda1']
    if args.lambda2 is None:
        args.lambda2 = db_config['lambda2']
    if args.n_groups is None:
        args.n_groups = db_config['n_groups']
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 80)
    print(f"TRAINING {args.model.upper()} MODEL")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir = output_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Database configuration: {args.database}")
    
    # Data transforms - 224×224 as per paper
    print("\nCreating data loaders...")
    print("📝 Paper Configuration:")
    print(f"   Input size: 224×224")
    print(f"   Batch size: {args.batch_size}")
    
    # Model-specific transforms
    # Paper Section IV-B: "the ROIs are resized to 224×224 pixels and duplicated to three channels"
    # RSNet: 3-channel (duplicated grayscale), EUSIPCO 2020: 3-channel RGB (for pretrained models)
    image_mode = 'L'  # Load as grayscale, will duplicate to 3 channels in transform
    
    if args.model == 'mpsnet':
        # MPSNet model configuration (matching original Keras dataloader)
        # Input: Grayscale (1 channel), Normalize to [0, 1] range
        # Original Keras: img = cv2.imread(path, 0); img_to_array(img)/255.0
        im_size = (224, 224)  # Standard size
        
        train_transform = transforms.Compose([
            transforms.Resize(im_size),           # Resize to 224x224
            BiometricCompose([
                CarefulRotation(max_angle=12.0, probability=0.8),
                CarefulTranslation(max_translate_percent=0.12, probability=0.8),
                GaussianNoise(std=0.025, probability=0.6),
                ContrastAdjustment(contrast_range=(0.75, 1.25), probability=0.7),
                BrightnessAdjustment(brightness_range=(0.88, 1.12), probability=0.7),
                RandomScale(scale_range=(0.92, 1.08), probability=0.6),
            ], overall_probability=0.98),
            transforms.Grayscale(num_output_channels=1),  # Keep as 1-channel grayscale
            transforms.ToTensor(),                # Converts to [0, 1] range automatically
            # NO normalization - matches Keras: img/255.0 which is exactly what ToTensor does
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(im_size),           # Resize to 224x224
            transforms.Grayscale(num_output_channels=1),  # Keep as 1-channel grayscale
            transforms.ToTensor(),                # Converts to [0, 1] range automatically
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(im_size),           # Resize to 224x224
            transforms.Grayscale(num_output_channels=1),  # Keep as 1-channel grayscale
            transforms.ToTensor(),                # Converts to [0, 1] range automatically
        ])
        
        print("📝 MPSNet Configuration (matching Keras dataloader):")
        print(f"   Input size: 224×224")
        print(f"   Channels: 1 (grayscale)")
        print(f"   Normalization: [0, 1] range (ToTensor only, no mean/std)")
        
    elif args.model == 'eusipco2020':
        # EUSIPCO 2020 model configuration
        # Image size: 228×228, Normalize to [-1, 1] range
        im_size = (228, 228)
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        train_transform = transforms.Compose([
            transforms.Resize(im_size),          # Resize to 228x228
            #transforms.ColorJitter(),            # (Commented) Color augmentation
            #transforms.RandomAffine(degrees=(-2,2), scale=(0.97,1.03), shear=(-2,2)),  # (Commented) Affine augmentation
            transforms.Grayscale(num_output_channels=3),  # Force 3 channels before ToTensor
            transforms.ToTensor(),                # Convert to tensor
            normalize                             # Normalize to [-1, 1] range
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(im_size),           # Resize to 228x228
            transforms.Grayscale(num_output_channels=3),  # Force 3 channels
            transforms.ToTensor(),                # Convert to tensor
            normalize                             # Normalize to [-1, 1] range
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(im_size),           # Resize to 228x228
            transforms.Grayscale(num_output_channels=3),  # Force 3 channels
            transforms.ToTensor(),                # Convert to tensor
            normalize                             # Normalize to [-1, 1] range
        ])
    elif args.model == 'rsnet':
        # RSNet: 3-channel (duplicated grayscale) - Paper Section IV-B
        # "the ROIs are resized to 224×224 pixels and duplicated to three channels"
        # 
        # Paper Data Augmentation (Section IV-B):
        # "the online data augmentation method, combining Random Perspective 
        # Transformation (RPT) and Random Gamma Adjustment (RGA) [6], is used"
        #
        # Parameters from AMPVNet [6] paper (referenced by RSNet):
        # - RPT: distortion_scale (r), probability (p_RPT)
        # - RGA: gamma_range (γ), probability (p_RGA)
        # Values are database-specific as per TABLE II of the paper
        
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Paper: 224×224
            BiometricCompose([
                # Paper-specified augmentations (Section IV-B)
                RandomPerspectiveTransformation(distortion_scale=0.15, probability=0.5),  # RPT: r=0.15, p=0.5
                RandomGammaAdjustment(gamma_range=(0.7, 1.3), probability=0.5),           # RGA: γ=(0.7,1.3), p=0.5
            ], overall_probability=1.0),  # Always apply augmentation pipeline
            transforms.Grayscale(num_output_channels=3),  # Paper: duplicate to 3 channels
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 3-channel normalization
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Paper: duplicate to 3 channels
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 3-channel normalization
        ])
        
        print("📝 RSNet Configuration (Paper Section IV-B):")
        print(f"   Input size: 224×224")
        print(f"   Channels: 3 (duplicated grayscale)")
        print(f"   Augmentation: RPT (r=0.15, p=0.5) + RGA (γ=0.7-1.3, p=0.5)")
    
    elif args.model == 'gscl':
        # GSCL (Generalized Supervised Contrastive Learning) configuration
        # Based on GSCL-PyTorch/vein_feature_learning/data/dataset.py get_transforms_sl() function
        im_size = (256, 256)
        
        image_mode = 'RGB'  # GSCL uses pretrained ImageNet models
        
        # Normalization to [-1, 1] as per GSCL paper
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        # GSCL augmentation pipeline from dataset.py:
        
        img_ratio = im_size[1] / im_size[0]
        train_transform = transforms.Compose([
            transforms.Resize(im_size),  # First resize to target
            transforms.RandomResizedCrop(size=im_size, scale=(0.5, 1.0), ratio=(img_ratio-0.5, img_ratio+0.5)),
            transforms.RandomRotation(degrees=3),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.9),
            transforms.ColorJitter(brightness=0.7, contrast=0.7),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel for pretrained ResNet
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print("📝 GSCL Configuration (from GSCL_2024/data/dataset.py):")
        print(f"   Input size: {im_size[0]}×{im_size[1]}")
        print(f"   Channels: 3 (for pretrained ResNet)")
        print(f"   Augmentation: RandomResizedCrop + Rotation(3°) + Perspective(0.3) + ColorJitter")
    
    elif args.model == 'sca_mobilenet_legacy':  # Legacy alias, use sca_mobilenet instead
        # MobileNetV3-Small + SPP + AdaCos
        # Uses ImageNet pretrained backbone - 3-channel RGB input
        im_size = (224, 224)
        image_mode = 'RGB'  # MobileNetV3 uses pretrained ImageNet model
        
        # ImageNet normalization for pretrained MobileNetV3
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        train_transform = transforms.Compose([
            transforms.Resize(im_size),
            BiometricCompose([
                CarefulRotation(max_angle=12.0, probability=0.8),
                CarefulTranslation(max_translate_percent=0.12, probability=0.8),
                GaussianNoise(std=0.025, probability=0.6),
                ContrastAdjustment(contrast_range=(0.75, 1.25), probability=0.7),
                BrightnessAdjustment(brightness_range=(0.88, 1.12), probability=0.7),
                RandomScale(scale_range=(0.92, 1.08), probability=0.6),
            ], overall_probability=0.98),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel for pretrained MobileNetV3
            transforms.ToTensor(),
            normalize
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize
        ])
        
        print("📝 MobileNetV3-Small + SPP Configuration:")
        print(f"   Input size: 224×224")
        print(f"   Channels: 3 (for pretrained MobileNetV3-Small)")
        print(f"   Normalization: ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])")
        print(f"   Pooling: SPP [1×1, 2×2, 4×4]")
        if args.loss_type == 'adacos_only':
            print(f"   Loss: AdaCos (Pure, no Triplet)")
        else:
            print(f"   Loss: FusionLoss ({args.loss_type.upper()} + Triplet)")
        print(f"   STN: DISABLED")
    
    elif args.model == 'sca_mobilenet':
        # SCA-MobileNet (Self-Contained Backbone)
        im_size = (224, 224)
        image_mode = 'RGB'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        train_transform = transforms.Compose([
            transforms.Resize(im_size),
            BiometricCompose([
                CarefulRotation(max_angle=12.0, probability=0.8),
                CarefulTranslation(max_translate_percent=0.12, probability=0.8),
                GaussianNoise(std=0.025, probability=0.6),
                ContrastAdjustment(contrast_range=(0.75, 1.25), probability=0.7),
                BrightnessAdjustment(brightness_range=(0.88, 1.12), probability=0.7),
                RandomScale(scale_range=(0.92, 1.08), probability=0.6),
            ], overall_probability=0.98),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize
        ])
        
        print("📝 SCA-MobileNet Configuration:")
        print(f"   Input size: 224×224")
        print(f"   Channels: 3 (ImageNet pretrained backbone)")
        print(f"   Backbone: {args.sca_backbone}")
        print(f"   Normalization: ImageNet")
        if args.loss_type == 'adacos_only':
            print(f"   Loss: AdaCos (Pure)")
        else:
            print(f"   Loss: FusionLoss ({args.loss_type.upper()} + Triplet)")
    
    elif args.model == 'mobilenetv3_uib':
        # MobileNetV3-Small + UIB + SPP + AdaCos
        # UIB: Universal Inverted Bottleneck from MobileNetV4
        im_size = (224, 224)
        image_mode = 'RGB'
        
        # ImageNet normalization for pretrained MobileNetV3
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        train_transform = transforms.Compose([
            transforms.Resize(im_size),
            BiometricCompose([
                CarefulRotation(max_angle=12.0, probability=0.8),
                CarefulTranslation(max_translate_percent=0.12, probability=0.8),
                GaussianNoise(std=0.025, probability=0.6),
                ContrastAdjustment(contrast_range=(0.75, 1.25), probability=0.7),
                BrightnessAdjustment(brightness_range=(0.88, 1.12), probability=0.7),
                RandomScale(scale_range=(0.92, 1.08), probability=0.6),
            ], overall_probability=0.98),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize
        ])
        
        print("📝 MobileNetV3-Small + UIB + SPP Configuration:")
        print(f"   Input size: 224×224")
        print(f"   Channels: 3 (for pretrained MobileNetV3-Small)")
        print(f"   Backbone: MobileNetV3-Small with UIB blocks (SE attention on all IR blocks)")
        print(f"   Normalization: ImageNet")
        print(f"   Pooling: SPP [1×1, 2×2, 4×4]")
        if args.loss_type == 'adacos_only':
            print(f"   Loss: AdaCos (Pure, no Triplet)")
        else:
            print(f"   Loss: FusionLoss ({args.loss_type.upper()} + Triplet)")
    
    elif args.model == 'fgfnet':
        # FGFNet (Palm-Vein-Spoof-Detection)
        # Input: 256x256 RGB
        im_size = (256, 256)
        image_mode = 'RGB'
        
        # Original: tf.image.per_image_standardization (mean=0, std=1 per image)
        # Here we use standard normalization as approximation or implement per-image standardization?
        # Standard: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] -> [-1, 1] range
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        train_transform = transforms.Compose([
            transforms.Resize(im_size),
            BiometricCompose([
                CarefulRotation(max_angle=12.0, probability=0.8),
                # CarefulTranslation(max_translate_percent=0.12, probability=0.8), # Spoof detection might be sensitive to translation?
                # GaussianNoise(std=0.025, probability=0.6),
                # ContrastAdjustment(contrast_range=(0.75, 1.25), probability=0.7),
                # BrightnessAdjustment(brightness_range=(0.88, 1.12), probability=0.7),
                RandomScale(scale_range=(0.92, 1.08), probability=0.6),
            ], overall_probability=0.8),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize
        ])
        
        print("📝 FGFNet Configuration (Palm-Vein-Spoof-Detection):")
        print(f"   Input size: 256×256")
        print(f"   Channels: 3 (RGB)")
        print(f"   Loss: CrossEntropy + 0.1 * Contrastive")
    
    # Load datasets
    train_dataset = PalmVeinDataset(
        Path(args.dataset) / 'train', 
        transform=train_transform,
        image_mode=image_mode
    )
    num_classes = len(train_dataset.class_to_idx)
    
    print(f"\nTraining set loaded:")
    print(f"  Classes: {num_classes}")
    print(f"  Samples: {len(train_dataset)}")
    
    # Load test dataset
    test_dataset_unfiltered = PalmVeinDataset(
        Path(args.dataset) / 'test', 
        transform=val_transform,
        valid_classes=None,
        image_mode=image_mode
    )
    
    # Check for open-set vs closed-set
    train_class_names = set(train_dataset.class_to_idx.keys())
    test_class_names = set(test_dataset_unfiltered.class_to_idx.keys())
    overlap = train_class_names & test_class_names
    is_open_set = len(overlap) == 0
    
    if is_open_set:
        print(f"\nOPEN-SET DETECTED (train and test have different users)")
        print(f"   - Validation loss will use train loss as proxy")
        print(f"   - Evaluation metrics (EER, TAR@FAR) will run on test users")
        test_dataset = None  # For validation loss calculation
    else:
        print(f"\nClosed-set dataset ({len(overlap)} overlapping users)")
        test_dataset = PalmVeinDataset(
            Path(args.dataset) / 'test',
            transform=val_transform,
            valid_classes=train_dataset.class_to_idx,
            image_mode=image_mode
        )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )
    
    # Test loader for validation loss (only for closed-set)
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda')
        )
    else:
        test_loader = None
    
    # Test loader for evaluation metrics (ALWAYS exists, even for open-set)
    test_loader_for_eval = DataLoader(
        test_dataset_unfiltered,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    # ==========================================================================
    # MODEL CREATION - Supports RSNet, EUSIPCO 2020, MPSNet, GSCL
    # ==========================================================================
    
    use_dual_branch = False  # Only RSNet uses dual branch
    use_gscl = False  # GSCL uses special training with FusionLoss
    use_fusion_v2 = False  # MobileNetV3 + FusionLoss (CosFace + Triplet)
    use_mobilenetv3_adacos = False  # MobileNetV3 + Pure AdaCos (no Triplet)
    use_fgfnet = False  # FGFNet: classification with aux contrastive loss
    is_transformer_backbone = False  # Transformer fine-tuning strategy
    transformer_base_lr = 2e-5
    transformer_lr_decay = 0.75
    
    if args.model == 'rsnet':
        # RSNet-ECA-RLEB (Paper Configuration)
        print(f"\nCreating RSNet model (Paper Configuration)...")
        print(f"  Feature dimension: {args.feature_dim}")
        print(f"  Dropout rate: {args.dropout}")
        print(f"  MAB subsets (d): {args.d}")
        print(f"  Channel shuffle groups (n): {args.n_groups}")
        
        model = RSNet(
            feature_dim=args.feature_dim,
            in_channels=3,  # Paper: "duplicated to three channels"
            dropout_rate=args.dropout,
            d=args.d,
            n_groups=args.n_groups
        ).to(device)
        
        use_dual_branch = True
        
        # Create AdaFace loss instances (paper configuration)
        adaface_loss_local = AdaFaceLoss(
            in_features=args.feature_dim,
            out_features=num_classes,
            scale=args.adaface_scale,
            margin=args.adaface_margin,
            h=args.adaface_h
        ).to(device)
        
        adaface_loss_global = AdaFaceLoss(
            in_features=args.feature_dim,
            out_features=num_classes,
            scale=args.adaface_scale,
            margin=args.adaface_margin,
            h=args.adaface_h
        ).to(device)
        
        joint_loss = RSNetJointLoss(
            adaface_loss_local=adaface_loss_local,
            adaface_loss_global=adaface_loss_global,
            lambda1=args.lambda1,
            lambda2=args.lambda2
        )
        
        print(f"\nLoss Configuration (AdaFace):")
        print(f"  Scale (s): {args.adaface_scale}")
        print(f"  Margin (m): {args.adaface_margin}")
        print(f"  Lambda1: {args.lambda1}, Lambda2: {args.lambda2}")
        
        # Optimizer
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(adaface_loss_local.parameters()) + list(adaface_loss_global.parameters()),
            lr=args.lr,
            betas=(0.9, 0.999)
        )
        
    elif args.model == 'eusipco2020':
        # EUSIPCO 2020 Models - using importlib for folder with spaces
        import importlib.util
        
        eusipco_dir = Path(__file__).parent / 'models' / 'Modified_Densenet161_2021'
        
        # Load modified_models module
        spec = importlib.util.spec_from_file_location(
            "modified_models", 
            eusipco_dir / 'models' / 'modified_models.py'
        )
        modified_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modified_models)
        
        DenseNet161_Modified = modified_models.DenseNet161_Modified
        Resnext101_32x8d_Modified = modified_models.Resnext101_32x8d_Modified
        MNASNet_Modified = modified_models.MNASNet_Modified
        
        backbone = args.eusipco_backbone
        print(f"\nCreating EUSIPCO 2020 model ({backbone})...")
        print(f"  Feature dimension: {args.feature_dim}")
        
        if backbone == 'densenet161':
            model = DenseNet161_Modified(
                embedding_size=args.feature_dim,
                class_size=num_classes,
                pretrained=True,
                only_embeddings=True,
                l2_normed=True
            ).to(device)
        elif backbone == 'resnext101':
            model = Resnext101_32x8d_Modified(
                embedding_size=args.feature_dim,
                class_size=num_classes,
                pretrained=True,
                only_embeddings=True,
                l2_normed=True
            ).to(device)
        else:  # mnasnet
            model = MNASNet_Modified(
                embedding_size=args.feature_dim,
                class_size=num_classes,
                pretrained=True,
                only_embeddings=True,
                l2_normed=True
            ).to(device)
        
        # EUSIPCO 2020 uses Cosine Loss (CosFace/LMCL) or ArcFace.
        # Load margin_losses module
        spec = importlib.util.spec_from_file_location(
            "margin_losses", 
            eusipco_dir / 'losses' / 'margin_losses.py'
        )
        margin_losses = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(margin_losses)
        
        AddMarginProduct = margin_losses.AddMarginProduct
        margin_loss = AddMarginProduct(
            in_features=args.feature_dim,
            out_features=num_classes,
            s=30.0,
            m=0.40
        ).to(device)
        
        ce_loss = nn.CrossEntropyLoss()
        
        print(f"\nLoss Configuration (CosFace/LMCL):")
        print(f"  Scale: 30.0, Margin: 0.40 (Paper Default)")
        
        # Optimizer
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(margin_loss.parameters()),
            lr=args.lr,
            betas=(0.9, 0.999)
        )
        
    elif args.model == 'mpsnet':
        # MPSNet (PyTorch Implementation)
        from MPSNet_2022.model_pytorch import MPSNet, AdaCos
        
        print(f"\nCreating MPSNet model (PyTorch Port)...")
        print(f"  Feature dimension: {args.feature_dim}")
        
        # MPSNet does not use pretrained weights, inputs are grayscale by default (can handle RGB)
        input_channels = 3 if args.model == 'eusipco2020' else 1
        
        # MPSNet logic: 1 channel input
        model = MPSNet(
            feature_dim=args.feature_dim,
            input_channels=1,
            dropout=args.dropout
        ).to(device)
        
        # MPSNet uses AdaCos
        margin_loss = AdaCos(
            num_features=args.feature_dim,
            num_classes=num_classes,
            m=0.50
        ).to(device)
        
        ce_loss = nn.CrossEntropyLoss()
        
        print(f"\nLoss Configuration (AdaCos):")
        print(f"  Dynamic Scale (simplified fixed s in PyTorch port)")
        
        # Optimizer
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(margin_loss.parameters()),
            lr=args.lr,
            betas=(0.9, 0.999)
        )

    elif args.model == 'gscl':
        # GSCL (Generalized Supervised Contrastive Learning)
        # Based on GSCL-PyTorch/vein_feature_learning
        import sys
        import os
        gscl_path = os.path.join(os.path.dirname(__file__), 'models', 'GSCL-PyTorch', 'vein_feature_learning')
        if gscl_path not in sys.path:
            sys.path.insert(0, gscl_path)
            
        from models.models import ResNets
        from loss.loss_functions import CosFace as GSCLCosFace, OnlineTripletLoss as GSCLOnlineTripletLoss, FusionLoss as GSCLFusionLoss
        
        print(f"\nCreating GSCL model ({args.gscl_backbone})...")
        print(f"  Feature dimension: {args.feature_dim}")
        print(f"  Backbone: {args.gscl_backbone}")
        
        # GSCL uses ResNet backbone with normalized linear head
        model = ResNets(
            backbone=args.gscl_backbone,
            head_type='cls_norm',  # Normalized linear for CosFace
            num_classes=num_classes
        ).to(device)
        
        # GSCL uses FusionLoss = w_cls * CosFace + w_metric * TripletLoss
        cosface_loss = GSCLCosFace(
            s=args.gscl_cosface_s,  # Default: 30.0
            m=args.gscl_cosface_m   # Default: 0.2
        )
        
        triplet_loss = GSCLOnlineTripletLoss(
            margin=args.gscl_triplet_margin,  # Default: 0.2
            is_distance=True
        )
        
        fusion_loss = GSCLFusionLoss(
            cls_loss=cosface_loss,
            metric_loss=triplet_loss,
            w_cls=args.gscl_w_cls,     # Default: 1.0
            w_metric=args.gscl_w_metric  # Default: 4.0
        )
        
        print(f"\nLoss Configuration (GSCL FusionLoss):")
        print(f"  CosFace: s={args.gscl_cosface_s}, m={args.gscl_cosface_m}")
        print(f"  TripletLoss: margin={args.gscl_triplet_margin}")
        print(f"  Weights: w_cls={args.gscl_w_cls}, w_metric={args.gscl_w_metric}")
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999)
        )
        
        # BalancedBatchSampler for triplet mining (need ≥2 samples per class)
        train_labels = [s[1] for s in train_dataset.samples]
        n_classes_per_batch = min(16, num_classes)
        n_samples_per_class = 4

        train_sampler = BalancedBatchSampler(
            train_labels,
            n_classes=n_classes_per_batch,
            n_samples=n_samples_per_class
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda')
        )

        print(f"\nDataLoader: BalancedBatchSampler")
        print(f"  {n_classes_per_batch} classes × {n_samples_per_class} samples = {n_classes_per_batch * n_samples_per_class} batch size")
        print(f"  Total batches per epoch: {len(train_sampler)}")

        # Mark as GSCL for special training logic
        use_gscl = True

    elif args.model == 'sca_mobilenet_legacy':  # Legacy alias, use sca_mobilenet instead
        # =====================================================================
        # MobileNetV3-Small + STN + CA + SPP (legacy)
        # =====================================================================
        use_stn = not args.no_stn   # STN configuration
        use_ca = not args.no_ca     # CA configuration
        use_spp = not args.no_spp   # SPP configuration
        
        print(f"\nCreating MobileNetV3 + STN + SPP model...")
        print(f"  Feature dimension: {args.feature_dim}")
        print(f"  Backbone: MobileNetV3-Small (pretrained)")
        print(f"  STN: {'Enabled' if use_stn else 'Disabled'}")
        print(f"  CA: {'Enabled' if use_ca else 'Disabled'}")
        print(f"  SPP: {'Enabled [1×1, 2×2, 4×4]' if use_spp else 'Disabled (using GAP)'}")
        print(f"  Dropout: {args.dropout}")
        
        # 1. Initialize Model
        model = SCAMobileNet(
            embedding_size=args.feature_dim,
            class_size=num_classes,
            pretrained=True,
            only_embeddings=True,
            use_stn=use_stn,
            use_ca=use_ca,
            use_spp=use_spp,
            dropout=args.dropout,
            use_bottleneck=not args.no_bottleneck,
            ca_reduction=args.ca_reduction
        ).to(device)

        # 2. Initialize Loss
        cls_type = args.loss_type  # 'cosface', 'adacos', or 'adacos_only'
        
        # Loss parameters
        scale_factor = 30.0
        # Reverted to 0.35 (Variant H best config)
        margin = 0.35 if cls_type == 'adacos_only' else 0.2
        triplet_margin = 0.2
        w_cls = 1.0
        w_metric = 4.0
        
        if cls_type == 'adacos_only':
            # Pure AdaCos loss (dynamic scale, fixed margin)
            margin_loss = AdaCosLoss(
                num_features=args.feature_dim,
                num_classes=num_classes,
                m=margin
            ).to(device)
            # Label smoothing for preventing overconfident predictions
            ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
            
            print(f"\nLoss Configuration:")
            print(f"  Type: AdaCos (Pure)")
            print(f"  AdaCos: dynamic s = sqrt(2)*log({num_classes}-1)")
            print(f"  Margin: {margin}")
            print(f"  CrossEntropyLoss with label_smoothing=0.1")
            
            # Optimizer: AdamW with weight decay (Single Group - Variant H)
            optimizer = torch.optim.AdamW(
                list(model.parameters()) + list(margin_loss.parameters()),
                lr=args.lr,
                betas=(0.9, 0.999), 
                weight_decay=1e-4
            )
            
            # BalancedBatchSampler for better convergence (same as FusionLoss)
            train_labels = [s[1] for s in train_dataset.samples]
            
            n_classes_per_batch = min(16, num_classes)
            n_samples_per_class = 4
            
            train_sampler = BalancedBatchSampler(
                train_labels, 
                n_classes=n_classes_per_batch,
                n_samples=n_samples_per_class
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=(device.type == 'cuda')
            )
            
            print(f"\nDataLoader: BalancedBatchSampler")
            print(f"  {n_classes_per_batch} classes × {n_samples_per_class} samples = {n_classes_per_batch * n_samples_per_class} batch size")
            print(f"  Total batches per epoch: {len(train_sampler)}")
            
            print(f"\nOptimizations Applied (Variant H):")
            print(f"  1. BalancedBatchSampler: diverse class distribution per batch")
            print(f"  2. Label Smoothing (0.1): prevents overconfident predictions")
            print(f"  3. AdamW with weight_decay=1e-4: regularization effect")
            print(f"  4. Unified LR: {args.lr} (No differential LR)")
            print(f"  5. Margin: {margin} (Optimal for Variant H)")
            
            # Mark as NOT using fusion strategy (uses single_branch training)
            use_fusion_v2 = False
            use_mobilenetv3_adacos = True
            
        else:
            # FusionLoss (CosFace/AdaCos + Triplet)
            fusion_criterion = FusionLoss(
                in_features=args.feature_dim,
                num_classes=num_classes,
                cls_type=cls_type,
                s=scale_factor,
                m=margin,
                triplet_margin=triplet_margin,
                w_cls=w_cls,
                w_metric=w_metric
            ).to(device)
            
            print(f"\nLoss Configuration:")
            print(f"  Type: FusionLoss ({cls_type.upper()} + Triplet)")
            if cls_type == 'cosface':
                print(f"  CosFace: s={scale_factor}, m={margin}")
            else:
                print(f"  AdaCos: dynamic s, m={margin}")
            print(f"  Triplet: margin={triplet_margin}")
            print(f"  Weights: w_cls={w_cls}, w_metric={w_metric}")
            
            # Optimizer: Standard Adam (same as other models)
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(fusion_criterion.parameters()),
                lr=args.lr,
                betas=(0.9, 0.999)
            )
            
            # 4. Modify DataLoader to use BalancedBatchSampler (CRITICAL FOR TRIPLET)
            train_labels = [s[1] for s in train_dataset.samples]
            
            n_classes_per_batch = min(16, num_classes)
            n_samples_per_class = 4
            
            train_sampler = BalancedBatchSampler(
                train_labels, 
                n_classes=n_classes_per_batch,
                n_samples=n_samples_per_class
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=(device.type == 'cuda')
            )
            
            print(f"\nDataLoader: BalancedBatchSampler")
            print(f"  {n_classes_per_batch} classes × {n_samples_per_class} samples = {n_classes_per_batch * n_samples_per_class} batch size")
            print(f"  Total batches per epoch: {len(train_sampler)}")
            
            # Mark as using fusion strategy
            use_fusion_v2 = True
            use_mobilenetv3_adacos = False

    elif args.model == 'sca_mobilenet':
        # =====================================================================
        # SCA-MobileNet (Self-Contained Backbone)
        # =====================================================================
        use_stn = not args.no_stn
        use_ca = not args.no_ca
        use_spp = not args.no_spp
        is_transformer_backbone = (args.sca_backbone != 'mobilenetv3')
        
        print(f"\nCreating SCA-MobileNet model...")
        print(f"  Feature dimension: {args.feature_dim}")
        print(f"  Backbone: {args.sca_backbone}")
        print(f"  STN: {'Enabled' if use_stn else 'Disabled'}")
        print(f"  CA: {'Enabled' if use_ca else 'Disabled'}")
        print(f"  SPP: {'Enabled [1×1, 2×2, 4×4]' if use_spp else 'Disabled (using GAP)'}")
        print(f"  Dropout: {args.dropout}")

        if args.sca_backbone == 'mobilenetv3':
            model = SCAMobileNet(
                embedding_size=args.feature_dim,
                class_size=num_classes,
                pretrained=True,
                only_embeddings=True,
                use_stn=use_stn,
                use_ca=use_ca,
                use_spp=use_spp,
                dropout=args.dropout,
                use_bottleneck=not args.no_bottleneck,
                ca_reduction=args.ca_reduction
            ).to(device)
        else:
            if args.no_stn or args.no_ca or args.no_spp:
                print("  Note: --no-stn/--no-ca/--no-spp are ignored for non-mobilenetv3 sca-backbone.")
            model = SCATransformerBackbone(
                backbone_name=args.sca_backbone,
                embedding_size=args.feature_dim,
                dropout=args.dropout,
                pretrained=True
            ).to(device)
        
        cls_type = args.loss_type

        if is_transformer_backbone:
            # Transformer backbones (Swin-Tiny, DeiT-Tiny, MobileViT-S) need:
            #   - Much lower LR to preserve pretrained weights
            #   - Gentler margin to avoid embedding collapse
            #   - Layer-wise LR decay so early layers change slowly
            #   - Warmup phase: freeze backbone for first 5 epochs
            transformer_base_lr = 2e-5
            transformer_lr_decay = 0.75
            margin = 0.10 if cls_type == 'adacos_only' else 0.10
            print(f"\n  Transformer fine-tuning strategy:")
            print(f"    Base LR: {transformer_base_lr} (layer-wise decay={transformer_lr_decay})")
            print(f"    Warmup: 5 epochs backbone-frozen, then full fine-tune")
            print(f"    Margin: {margin}")
        else:
            margin = 0.35 if cls_type == 'adacos_only' else 0.2
        
        if cls_type == 'adacos_only':
            margin_loss = AdaCosLoss(
                num_features=args.feature_dim,
                num_classes=num_classes,
                m=margin
            ).to(device)
            ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
            
            if is_transformer_backbone:
                # Layer-wise LR decay for transformer backbone
                model.freeze_backbone()
                param_groups = model.get_layer_wise_param_groups(
                    base_lr=transformer_base_lr,
                    decay=transformer_lr_decay
                )
                param_groups.append({
                    'params': list(margin_loss.parameters()),
                    'lr': transformer_base_lr
                })
                optimizer = torch.optim.AdamW(
                    param_groups,
                    betas=(0.9, 0.999),
                    weight_decay=5e-2
                )
            else:
                optimizer = torch.optim.AdamW(
                    list(model.parameters()) + list(margin_loss.parameters()),
                    lr=args.lr,
                    betas=(0.9, 0.999),
                    weight_decay=1e-4
                )
            
            train_labels = [s[1] for s in train_dataset.samples]
            n_classes_per_batch = min(16, num_classes)
            n_samples_per_class = 4
            
            train_sampler = BalancedBatchSampler(
                train_labels,
                n_classes=n_classes_per_batch,
                n_samples=n_samples_per_class
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=(device.type == 'cuda')
            )
            
            print(f"\nLoss: AdaCos (margin={margin}) + CE (label_smoothing=0.1)")
            print(f"DataLoader: BalancedBatchSampler ({n_classes_per_batch}×{n_samples_per_class})")
            
            use_fusion_v2 = False
            use_mobilenetv3_adacos = True
        else:
            if is_transformer_backbone:
                margin = 0.10
            fusion_criterion = FusionLoss(
                in_features=args.feature_dim,
                num_classes=num_classes,
                cls_type=cls_type,
                s=30.0,
                m=margin,
                triplet_margin=0.2,
                w_cls=1.0,
                w_metric=4.0
            ).to(device)
            
            if is_transformer_backbone:
                model.freeze_backbone()
                param_groups = model.get_layer_wise_param_groups(
                    base_lr=transformer_base_lr,
                    decay=transformer_lr_decay
                )
                param_groups.append({
                    'params': list(fusion_criterion.parameters()),
                    'lr': transformer_base_lr
                })
                optimizer = torch.optim.AdamW(
                    param_groups,
                    betas=(0.9, 0.999),
                    weight_decay=5e-2
                )
            else:
                optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(fusion_criterion.parameters()),
                    lr=args.lr,
                    betas=(0.9, 0.999)
                )
            
            train_labels = [s[1] for s in train_dataset.samples]
            n_classes_per_batch = min(16, num_classes)
            n_samples_per_class = 4
            
            train_sampler = BalancedBatchSampler(
                train_labels,
                n_classes=n_classes_per_batch,
                n_samples=n_samples_per_class
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=(device.type == 'cuda')
            )
            
            use_fusion_v2 = True
            use_mobilenetv3_adacos = False

    elif args.model == 'mobilenetv3_uib':
        # =====================================================================
        # MobileNetV3-Small + UIB (Universal Inverted Bottleneck) + CA + SPP
        # UIB brings MobileNetV4-style SE attention to all InvertedResidual blocks
        # =====================================================================
        use_stn = True   # STN configuration (ENABLED)
        use_ca = True   # CA configuration (ENABLED)
        use_uib = True   # UIB configuration (ENABLED)
        
        print(f"\nCreating MobileNetV3 + UIB + STN + SPP model...")
        print(f"  Feature dimension: {args.feature_dim}")
        print(f"  Backbone: MobileNetV3-Small with UIB blocks (pretrained)")
        print(f"  UIB: Enabled (SE attention on all IR blocks)")
        print(f"  STN: {'Enabled' if use_stn else 'Disabled'}")
        print(f"  CA: {'Enabled' if use_ca else 'Disabled'}")
        print(f"  SPP: Enabled [1×1, 2×2, 4×4]")
        print(f"  Dropout: {args.dropout}")
        
        # 1. Initialize Model with UIB
        model = MobileNetV3_UIB_SPP(
            embedding_size=args.feature_dim,
            class_size=num_classes,
            pretrained=True,
            only_embeddings=True,
            use_stn=use_stn,
            use_ca=use_ca,
            use_uib=use_uib,
            dropout=args.dropout
        ).to(device)
        
        # 2. Loss Configuration (same as sca_mobilenet)
        cls_type = args.loss_type  # 'cosface', 'adacos', or 'adacos_only'
        
        scale_factor = 30.0
        # REDUCED margin: 0.35 caused embedding collapse, 0.1 is safer
        margin = 0.1 if cls_type == 'adacos_only' else 0.25
        triplet_margin = 0.3
        w_cls = 1.0
        w_metric = 2.0
        
        if cls_type == 'adacos_only':
            # Pure AdaCos loss (dynamic scale, fixed margin)
            margin_loss = AdaCosLoss(
                num_features=args.feature_dim,
                num_classes=num_classes,
                m=margin
            ).to(device)
            
            ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
            
            print(f"\nLoss Configuration:")
            print(f"  Type: AdaCos (Pure)")
            print(f"  AdaCos: dynamic s = sqrt(2)*log({num_classes}-1)")
            print(f"  Margin: {margin}")
            print(f"  CrossEntropyLoss with label_smoothing=0.1")
            
            optimizer = torch.optim.AdamW(
                list(model.parameters()) + list(margin_loss.parameters()),
                lr=args.lr,
                betas=(0.9, 0.999),
                weight_decay=1e-4
            )
            
            train_labels = [s[1] for s in train_dataset.samples]
            
            n_classes_per_batch = min(16, num_classes)
            n_samples_per_class = 4
            
            train_sampler = BalancedBatchSampler(
                train_labels, 
                n_classes=n_classes_per_batch,
                n_samples=n_samples_per_class
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=(device.type == 'cuda')
            )
            
            print(f"\nDataLoader: BalancedBatchSampler")
            print(f"  {n_classes_per_batch} classes × {n_samples_per_class} samples = {n_classes_per_batch * n_samples_per_class} batch size")
            print(f"  Total batches per epoch: {len(train_sampler)}")
            
            # Mark as NOT using fusion strategy (uses single_branch training)
            use_fusion_v2 = False
            use_mobilenetv3_adacos = True
            
        else:
            # FusionLoss (CosFace/AdaCos + Triplet)
            fusion_criterion = FusionLoss(
                in_features=args.feature_dim,
                num_classes=num_classes,
                cls_type=cls_type,
                s=scale_factor,
                m=margin,
                triplet_margin=triplet_margin,
                w_cls=w_cls,
                w_metric=w_metric
            ).to(device)
            
            print(f"\nLoss Configuration:")
            print(f"  Type: FusionLoss ({cls_type.upper()} + Triplet)")
            if cls_type == 'cosface':
                print(f"  CosFace: s={scale_factor}, m={margin}")
            else:
                print(f"  AdaCos: dynamic s, m={margin}")
            print(f"  Triplet: margin={triplet_margin}")
            print(f"  Weights: w_cls={w_cls}, w_metric={w_metric}")
            
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(fusion_criterion.parameters()),
                lr=args.lr,
                betas=(0.9, 0.999)
            )
            
            train_labels = [s[1] for s in train_dataset.samples]
            
            n_classes_per_batch = min(16, num_classes)
            n_samples_per_class = 4
            
            train_sampler = BalancedBatchSampler(
                train_labels, 
                n_classes=n_classes_per_batch,
                n_samples=n_samples_per_class
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=(device.type == 'cuda')
            )
            
            print(f"\nDataLoader: BalancedBatchSampler")
            print(f"  {n_classes_per_batch} classes × {n_samples_per_class} samples = {n_classes_per_batch * n_samples_per_class} batch size")
            print(f"  Total batches per epoch: {len(train_sampler)}")
            
            # Mark as using fusion strategy
            use_fusion_v2 = True
            use_mobilenetv3_adacos = False
        
    elif args.model == 'fgfnet':
        # FGFNet Model
        print(f"\nCreating FGFNet (MobileViT_FFC_ATTN_FFTSA)...")
        
        model = MobileViT_FFC_ATTN_FFTSA(
            image_size=(256, 256),
            num_classes=num_classes
        ).to(device)
        
        # FGFNet Loss
        criterion = FGFNetLoss(
            contrastive_weight=0.1,  # From TF: 0.1 * contrastive_loss
            temperature=0.1
        ).to(device)
        
        print(f"\nLoss Configuration (FGFNet):")
        print(f"  CrossEntropy + 0.1 * ContrastiveLoss(temp=0.1)")
        
        # Note: TF used lr=0.00001 (1e-5). 
        fgfnet_lr = args.lr if args.lr != 0.001 else 1e-4
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=fgfnet_lr
        )
        print(f"  Optimizer: Adam (lr={fgfnet_lr})")

        use_fgfnet = True

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Scheduler selection
    use_transformer_schedule = is_transformer_backbone

    if use_transformer_schedule:
        # Cosine annealing with linear warmup works much better for transformers
        warmup_epochs = 5
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - warmup_epochs,
            eta_min=1e-7
        )
        print(f"Scheduler: CosineAnnealingLR (T_max={args.epochs - warmup_epochs}, eta_min=1e-7)")
        print(f"  Warmup: {warmup_epochs} epochs (backbone frozen)")
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[30, 60, 85],
            gamma=0.1
        )
        print("Scheduler: MultiStepLR (Decay at 30, 60, 85)")
    
    print(f"\nOptimizer Configuration:")
    if use_transformer_schedule:
        print(f"  Optimizer: AdamW (layer-wise LR decay)")
        print(f"  Base LR: {transformer_base_lr}")
        print(f"  Scheduler: CosineAnnealingLR")
    else:
        print(f"  Optimizer: Adam")
        print(f"  Initial LR: {args.lr}")
        print(f"  Scheduler: MultiStepLR (Decay at 30, 60, 85)")
    
    monitor = TrainingMonitor(
        log_dir=str(output_dir / 'logs'),
        save_plots=True,
        plot_frequency=5
    )
    
    # Early stopping (optional - NOT used in paper by default)
    # Paper strategy: run full 100 epochs, evaluate every 5 epochs, select best EER manually
    early_stopping = None
    if args.use_early_stopping:
        early_stopping = EarlyStopping(
            patience=args.early_stop_patience,
            min_delta=0.001,
            monitor='val_loss',  # Monitor validation loss
            mode='min',
            restore_best_weights=False,  # We save best model manually
            save_best_model=False,  # We handle saving manually
            verbose=1
        )
        print(f"\n⚠️  Early stopping ENABLED (patience={args.early_stop_patience})")
        print(f"   Note: Paper does NOT use early stopping - runs full {args.epochs} epochs")
    else:
        print(f"\n✓ Paper strategy: Run full {args.epochs} epochs, evaluate every {args.eval_frequency} epochs")
    
    scaler = GradScaler()
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Evaluation strategy: Test every {args.eval_frequency} epochs")
    print("=" * 80)
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # --- WARM-UP LOGIC ---
        if use_transformer_schedule:
            # Transformer backbone warmup: freeze backbone for first 5 epochs
            if epoch == 1:
                model.freeze_backbone()
                print("  >> Transformer warmup: backbone frozen, training head + loss only")
            elif epoch == 6:
                model.unfreeze_backbone()
                # Rebuild optimizer param groups with layer-wise LR decay now that backbone is unfrozen
                param_groups = model.get_layer_wise_param_groups(
                    base_lr=transformer_base_lr,
                    decay=transformer_lr_decay
                )
                if use_mobilenetv3_adacos:
                    param_groups.append({
                        'params': list(margin_loss.parameters()),
                        'lr': transformer_base_lr
                    })
                else:
                    param_groups.append({
                        'params': list(fusion_criterion.parameters()),
                        'lr': transformer_base_lr
                    })
                optimizer = torch.optim.AdamW(
                    param_groups,
                    betas=(0.9, 0.999),
                    weight_decay=5e-2
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=args.epochs - 5,
                    eta_min=1e-7
                )
                scaler = GradScaler()
                trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  >> Transformer fine-tune: ALL layers unfrozen ({trainable_now:,} params)")
                print(f"     Optimizer rebuilt with layer-wise LR decay")
        elif use_fusion_v2:
            # MobileNetV3 FusionLoss warmup (original logic)
            if epoch == 1:
                set_training_mode(model, 'warmup')
                print("  >> Warmup phase: Training CA, Head & Loss only (backbone frozen)")
            elif epoch == 6:
                set_training_mode(model, 'finetune')
                print("  >> Fine-tune phase: Training entire model")
        
        # --- SELECT TRAINING FUNCTION ---
        if use_dual_branch:  # RSNet
            train_losses = train_epoch_rsnet(
                model, train_loader, joint_loss, optimizer, scaler, device, epoch
            )
        elif use_gscl:  # GSCL
            train_losses = train_epoch_gscl(
                model, train_loader, fusion_loss, optimizer, scaler, device, epoch
            )
        elif use_fusion_v2:  # MobileNetV3 + FusionLoss
            train_losses = train_epoch_fusion_v2(
                model, train_loader, fusion_criterion, optimizer, scaler, device, epoch
            )
        elif use_fgfnet:
            # FGFNet (Spoof Detection)
            train_losses = train_epoch_fgfnet(
                model, train_loader, criterion, optimizer, scaler, device, epoch
            )
        else:  # EUSIPCO type or MPSNet (single branch)
            train_losses = train_epoch_single_branch(
                model, train_loader, margin_loss, ce_loss, optimizer, scaler, device, epoch,
                model_type=args.model
            )
        
        # Evaluate on test set only every N epochs (paper: every 5)
        should_evaluate = (epoch % args.eval_frequency == 0) or (epoch == args.epochs)
        
        if should_evaluate:
            # Validation loss
            if test_loader is not None:
                if use_dual_branch:  # RSNet
                    val_losses = validate_rsnet(model, test_loader, joint_loss, device)
                    
                    monitor.log_epoch(epoch, train_losses['total'], {
                        'val_loss': val_losses['val_loss'],
                        'val_loss_local': val_losses['val_loss_local'],
                        'val_loss_global': val_losses['val_loss_global'],
                        'val_loss_diff': val_losses['val_loss_diff'],
                        'train_loss_local': train_losses['local'],
                        'train_loss_global': train_losses['global'],
                        'train_loss_diff': train_losses['diff']
                    }, optimizer.param_groups[0]['lr'])
                    
                    print(f"\n[{time.strftime('%H:%M:%S')}] Epoch {epoch:3d}/{args.epochs}")
                    print(f"  Train - Total: {train_losses['total']:.4f} | "
                          f"Local: {train_losses['local']:.4f} | "
                          f"Global: {train_losses['global']:.4f} | "
                          f"Diff: {train_losses['diff']:.6f}")
                    print(f"  Test  - Total: {val_losses['val_loss']:.4f} | "
                          f"Local: {val_losses['val_loss_local']:.4f} | "
                          f"Global: {val_losses['val_loss_global']:.4f} | "
                          f"Diff: {val_losses['val_loss_diff']:.6f}")
                    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
                elif use_gscl:  # GSCL
                    val_losses = validate_gscl(model, test_loader, fusion_loss, device)
                    
                    monitor.log_epoch(epoch, train_losses['total'], {
                        'val_loss': val_losses['val_loss']
                    }, optimizer.param_groups[0]['lr'])
                    
                    print(f"\n[{time.strftime('%H:%M:%S')}] Epoch {epoch:3d}/{args.epochs}")
                    print(f"  Train Loss: {train_losses['total']:.4f}")
                    print(f"  Val Loss:   {val_losses['val_loss']:.4f}")
                    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

                    
                    monitor.log_epoch(epoch, train_losses['total'], {
                        'val_loss': val_losses['val_loss'],
                        'val_acc': val_losses['val_acc']
                    }, optimizer.param_groups[0]['lr'])
                    
                    print(f"\n[{time.strftime('%H:%M:%S')}] Epoch {epoch:3d}/{args.epochs}")
                    print(f"  Train Loss: {train_losses['total']:.4f} | Train Acc: {train_losses['acc']:.2f}%")
                    print(f"  Val Loss:   {val_losses['val_loss']:.4f} | Val Acc:   {val_losses['val_acc']:.2f}%")
                    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
                elif use_fgfnet:
                    val_losses = validate_fgfnet(model, test_loader, criterion, device)
                    
                    monitor.log_epoch(epoch, train_losses['total'], {
                        'val_loss': val_losses['val_loss'],
                        'val_acc': val_losses['val_acc']
                    }, optimizer.param_groups[0]['lr'])
                    
                    print(f"\n[{time.strftime('%H:%M:%S')}] Epoch {epoch:3d}/{args.epochs}")
                    print(f"  Train Loss: {train_losses['total']:.4f} (CE: {train_losses['ce']:.4f})")
                    print(f"  Val Loss:   {val_losses['val_loss']:.4f} | Val Acc:   {val_losses['val_acc']:.2f}%")
                    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
                elif use_fusion_v2:  # MobileNetV3 + FusionLoss
                    val_losses = validate_fusion_v2(model, test_loader, fusion_criterion, device)
                    
                    monitor.log_epoch(epoch, train_losses['total'], {
                        'val_loss': val_losses['val_loss']
                    }, optimizer.param_groups[0]['lr'])
                    
                    print(f"\n[{time.strftime('%H:%M:%S')}] Epoch {epoch:3d}/{args.epochs}")
                    print(f"  Train Loss: {train_losses['total']:.4f}")
                    print(f"  Val Loss:   {val_losses['val_loss']:.4f}")
                    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
                    print(f"  Metric Weight: {fusion_criterion.w_metric:.1f}")
                else:  # Single branch (EUSIPCO/MPSNet)
                    val_losses = validate_single_branch(model, test_loader, margin_loss, ce_loss, device, model_type=args.model)
                    
                    monitor.log_epoch(epoch, train_losses['total'], {
                        'val_loss': val_losses['val_loss']
                    }, optimizer.param_groups[0]['lr'])
                    
                    print(f"\n[{time.strftime('%H:%M:%S')}] Epoch {epoch:3d}/{args.epochs}")
                    print(f"  Train Loss: {train_losses['total']:.4f}")
                    print(f"  Val Loss:   {val_losses['val_loss']:.4f}")
                    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            else:
                # Open-set: No validation loss
                if use_dual_branch:
                    monitor.log_epoch(epoch, train_losses['total'], {
                        'train_loss_local': train_losses['local'],
                        'train_loss_global': train_losses['global'],
                        'train_loss_diff': train_losses['diff']
                    }, optimizer.param_groups[0]['lr'])
                    
                    print(f"\n[{time.strftime('%H:%M:%S')}] Epoch {epoch:3d}/{args.epochs}")
                    print(f"  Train - Total: {train_losses['total']:.4f} | "
                          f"Local: {train_losses['local']:.4f} | "
                          f"Global: {train_losses['global']:.4f} | "
                          f"Diff: {train_losses['diff']:.6f}")
                    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
                elif use_fgfnet:
                    monitor.log_epoch(epoch, train_losses['total'], {}, optimizer.param_groups[0]['lr'])
                    print(f"\n[{time.strftime('%H:%M:%S')}] Epoch {epoch:3d}/{args.epochs}")
                    print(f"  Train Loss: {train_losses['total']:.4f}")
                    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
                elif use_fusion_v2:
                    monitor.log_epoch(epoch, train_losses['total'], {}, optimizer.param_groups[0]['lr'])
                    
                    print(f"\n[{time.strftime('%H:%M:%S')}] Epoch {epoch:3d}/{args.epochs}")
                    print(f"  Train Loss: {train_losses['total']:.4f}")
                    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
                    print(f"  Metric Weight: {fusion_criterion.w_metric:.1f}")
                else:
                    monitor.log_epoch(epoch, train_losses['total'], {}, optimizer.param_groups[0]['lr'])
                    
                    print(f"\n[{time.strftime('%H:%M:%S')}] Epoch {epoch:3d}/{args.epochs}")
                    print(f"  Train Loss: {train_losses['total']:.4f}")
                    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Evaluate verification metrics (Paper: EER, TAR@FAR every 5 epochs)
            metrics = evaluate_verification_metrics(
                model, test_loader_for_eval, device, epoch,
                balanced_sampling=True,  # Paper strategy: 1:1 genuine-imposter ratio
                model_type=args.model
            )
            
            # Log metrics
            print(f"\n  Key Metrics:")
            print(f"     EER: {metrics['eer']*100:.4f}% | AUC: {metrics['auc']:.6f} | D': {metrics['d_prime']:.4f}")
            
            # Save best model based on EER (paper's primary metric)
            if 'best_eer' not in locals():
                best_eer = float('inf')
            
            if metrics['eer'] < best_eer:
                best_eer = metrics['eer']
                print(f"  >> New best EER: {best_eer*100:.4f}% (epoch {epoch})")
                
                # Build config based on model type
                if use_dual_branch:  # RSNet
                    config = {
                        'model_type': 'rsnet',
                        'feature_dim': args.feature_dim,
                        'dropout': args.dropout,
                        'd': args.d,
                        'n_groups': args.n_groups,
                        'lambda1': args.lambda1,
                        'lambda2': args.lambda2,
                        'adaface_scale': args.adaface_scale,
                        'adaface_margin': args.adaface_margin,
                        'adaface_h': args.adaface_h,
                        'num_classes': num_classes,
                        'database': args.database
                    }
                elif use_gscl:  # GSCL
                    config = {
                        'model_type': 'gscl',
                        'feature_dim': args.feature_dim,
                        'backbone': args.gscl_backbone,
                        'num_classes': num_classes,
                        'loss': 'FusionLoss (CosFace + TripletLoss)',
                        'cosface_s': args.gscl_cosface_s,
                        'cosface_m': args.gscl_cosface_m,
                        'triplet_margin': args.gscl_triplet_margin,
                        'w_cls': args.gscl_w_cls,
                        'w_metric': args.gscl_w_metric
                    }
                elif use_fusion_v2:  # MobileNetV3 + STN + CA + SPP + FusionLoss
                    config = {
                        'model_type': 'sca_mobilenet',
                        'feature_dim': args.feature_dim,
                        'num_classes': num_classes,
                        'backbone': 'MobileNetV3-Small (pretrained)',
                        'use_stn': True,
                        'use_spp': True,
                        'use_ca': True,
                        'dropout': args.dropout,
                        'loss': 'FusionLoss (CosFace + Triplet)',
                        'cosface_s': 30.0,
                        'cosface_m': 0.20,
                        'triplet_margin': 0.3,
                        'w_cls': 1.0,
                        'w_metric': 0.5
                    }
                else:  # EUSIPCO 2020, MPSNet
                    config = {
                        'model_type': args.model,
                        'feature_dim': args.feature_dim,
                        'num_classes': num_classes
                    }
                    if args.model == 'eusipco2020':
                        config['backbone'] = args.eusipco_backbone
                        config['loss'] = 'CosFace (AddMargin)'
                    elif args.model == 'mpsnet':
                        config['loss'] = 'AdaCos'
                        config['dropout'] = args.dropout
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config,
                    'metrics': metrics,
                    'eer': metrics['eer']
                }, output_dir / f'best_{run_name}_model_eer.pth')
            
            # Save metrics to JSON for analysis
            metrics_to_save = {
                **metrics,
                'train_loss': train_losses['total'],
                'test_loss': val_losses['val_loss'] if test_loader is not None else train_losses['total']
            }
            save_metrics_to_json(metrics_to_save, output_dir, epoch)
            
            # Also keep best loss model (backward compatibility)
            if test_loader is not None and val_losses['val_loss'] < best_val_loss:
                best_val_loss = val_losses['val_loss']
                print(f"  >> New best test loss: {best_val_loss:.4f}")
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': {
                        'feature_dim': args.feature_dim,
                        'dropout': args.dropout,
                        'd': args.d,
                        'n_groups': args.n_groups,
                        'lambda1': args.lambda1,
                        'lambda2': args.lambda2,
                        'adaface_scale': args.adaface_scale,
                        'adaface_margin': args.adaface_margin,
                        'adaface_h': args.adaface_h,
                        'num_classes': num_classes,
                        'database': args.database
                    },
                    'val_loss': val_losses['val_loss']
                }, output_dir / 'best_model.pth')
            
            # Check early stopping (if enabled)
            if early_stopping is not None:
                should_stop = early_stopping(epoch, val_losses, model)
                if should_stop:
                    print(f"\n  Early stopping triggered at epoch {epoch}")
                    print(f"   Best epoch: {early_stopping.get_best_epoch()}")
                    print(f"   Best val_loss: {early_stopping.get_best_score():.4f}")
                    break
        else:
            # No evaluation this epoch - just training
            if use_dual_branch:  # RSNet has local/global/diff losses
                monitor.log_epoch(epoch, train_losses['total'], {
                    'train_loss_local': train_losses['local'],
                    'train_loss_global': train_losses['global'],
                    'train_loss_diff': train_losses['diff']
                }, optimizer.param_groups[0]['lr'])
            else:
                # Single-branch models only have total loss
                monitor.log_epoch(epoch, train_losses['total'], {}, optimizer.param_groups[0]['lr'])
            
            # Simplified log
            next_eval = ((epoch // args.eval_frequency) + 1) * args.eval_frequency
            print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {train_losses['total']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Next eval: {next_eval}")
        
        # Update learning rate (skip during transformer warmup phase)
        if use_transformer_schedule and epoch <= 5:
            pass  # Don't step scheduler during warmup
        else:
            scheduler.step()
        
        # Save checkpoint every eval_frequency epochs (disable with --no-checkpoint)
        if not args.no_checkpoint and epoch % args.eval_frequency == 0:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            if should_evaluate and test_loader is not None:
                checkpoint_data['val_loss'] = val_losses['val_loss']
            torch.save(checkpoint_data, checkpoints_dir / f'checkpoint_epoch_{epoch}.pth')
    
    # Training completed
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Best test loss: {best_val_loss:.4f}")
    print(f"\nModel saved to: {output_dir / 'best_model.pth'}")
    print(f"Logs saved to: {output_dir / 'logs'}")
    
    # Generate visualization charts
    print("\n Generating visualization charts...")
    try:
        generate_training_charts(
            output_dir=str(output_dir),
            genuine_scores=None,  # Will load from training_metrics.json
            imposter_scores=None,
            eer_threshold=None
        )
        print(f"Charts saved to: {output_dir / 'charts'}")
    except Exception as e:
        print(f"Warning: Could not generate charts: {e}")
    
if __name__ == "__main__":
    main()
