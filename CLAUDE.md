# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Palm vein biometric authentication research platform. Implements multiple deep learning architectures for palm vein recognition with comprehensive evaluation protocols (EER, TAR@FAR, AUC, D-prime). The proposed model is **SCA-MobileNet**, benchmarked against baselines across multiple datasets. Codebase comments contain a mix of English and Vietnamese.

**Language**: Python 3.9+ / **Framework**: PyTorch 2.0+ / **Mixed precision**: `torch.cuda.amp`

## Project Structure

```
PalmVein/
├── train.py                    # Central training entry point (all models)
├── CLAUDE.md / INSTALL.md / requirements.txt
│
├── models/                     # All model architectures & shared libraries
│   ├── SCA_MobileNet/          #   Proposed model (STN + CoordAttention + SPP)
│   ├── RSNet/                  #   RSNet + shared losses (AdaFace)
│   ├── biometric/              #   Metrics, augmentation, visualization
│   ├── FGFNet/                 #   FGFNet baseline (MobileViT + FFC)
│   ├── VeinKAN/                #   VeinKAN baseline (InceptionV3 + KAN)
│   ├── GSCL_2024/              #   GSCL 2024 (ResNet + contrastive)
│   ├── MPSNet_2022/            #   MPSNet baseline
│   ├── Modified_Densenet161_2021/  # Modified DenseNet-161
│   ├── GSCL-PyTorch/           #   GSCL git clone (separate conda env)
│   └── Palm-Vein-Spoof-Detection/  # Spoof detection (TensorFlow)
│
├── preprocessing/              # Data preparation pipeline
│   ├── palm_vein_preprocessing.py  # GrabCut → ROI → 128×128
│   ├── palm_vein_enhancement.py    # CLAHE enhancement
│   ├── prepare_*_dataset.py        # Dataset-specific preparation
│   ├── split_*_openset.py          # Identity-level 70/30 splits
│   └── setup_internal_dataset.py   # Internal dataset setup
│
├── evaluation/                 # Analysis & evaluation scripts
│   ├── cross_domain_eval.py    # Cross-domain evaluation
│   ├── gradcam_visualization.py
│   ├── inference_benchmark.py
│   ├── analyze_rejections.py
│   └── statistical_significance.py
│
├── datasets/                   # All prepared datasets
├── results/                    # All training results (70+ dirs)
├── scripts/                    # Batch scripts (.bat + .sh)
├── docs/                       # Documentation, diagrams, PDFs
├── paper/                      # LaTeX paper workspace
├── pretrained/                 # Pretrained model weights
└── tools/                      # One-off utility scripts
```

## Setup

```bash
# Windows
scripts/setup.bat

# Linux
chmod +x scripts/setup.sh && scripts/setup.sh

# Manual
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate.bat on Windows
pip install -r requirements.txt
```

If you get `ModuleNotFoundError: No module named 'biometric'`, set `PYTHONPATH` to include the project root.

## Common Commands

### Data Preparation Pipeline

Raw images → preprocessing → enhancement → open-set split → training:

```bash
# 1. Preprocessing (GrabCut segmentation → orientation → ROI → resize 128×128)
python preprocessing/palm_vein_preprocessing.py

# 2. CLAHE enhancement
python preprocessing/palm_vein_enhancement.py

# 3. Dataset-specific preparation
python preprocessing/prepare_scut_dataset.py      # → SCUT_enhanced/
python preprocessing/prepare_tongji_dataset.py    # → TONGJI_enhanced/
python preprocessing/prepare_vera_dataset.py      # → VERA_enhanced/

# 4. Open-set split (identity-level disjoint 70/30 train/test)
python preprocessing/split_scut_openset.py        # → datasets/SCUT_dataset_openset/
python preprocessing/split_tongji_openset.py      # → datasets/TONGJI_dataset_openset/
python preprocessing/split_vera_openset.py        # → datasets/VERA_dataset_openset/

# Closed-set split (per-user 70/30)
python preprocessing/dataset_splitter.py --input roi_dataset --output final_dataset
```

### Training

```bash
python train.py --model rsnet --dataset datasets/TONGJI_dataset_openset
python train.py --model sca_mobilenet --dataset datasets/SCUT_dataset_openset
python train.py --model fgfnet --dataset datasets/VERA_dataset_openset

# Key training flags
python train.py --model sca_mobilenet --dataset datasets/SCUT_dataset_openset \
    --batch-size 16 --epochs 100 --lr 0.001 \
    --feature-dim 1024 --eval-frequency 5 \
    --database SCUT \
    --loss-type adacos_only \
    --sca-backbone mobilenetv3

# Available --model: rsnet | eusipco2020 | mpsnet | gscl | sca_mobilenet | mobilenetv3_uib | fgfnet
# Note: veinkan is handled in code but not in argparse choices

# SCA-MobileNet backbone alternatives (--sca-backbone):
# mobilenetv3 (default) | mobilevit_s | deit_tiny | swin_tiny | efficientnet_b0

# SCA-MobileNet loss types (--loss-type):
# cosface | adacos | adacos_only (default)

# SCA-MobileNet ablation flags:
# --no-stn   Disable Spatial Transformer Network
# --no-ca    Disable CoordAttention
# --no-spp   Disable Spatial Pyramid Pooling
# (These flags are ignored for non-mobilenetv3 sca-backbones)

# --database: CASIA_850 | CASIA_940 | VERA | TJ_PV | PLUS_PV850 | PLUS_PV950 | SCUT | default
```

### Evaluation

```bash
python evaluation/cross_domain_eval.py --model sca_mobilenet   # Train on DTS, test on TONGJI
python evaluation/statistical_significance.py                    # Wilcoxon tests across 5 seeds (42, 0, 1, 7, 99)
python evaluation/analyze_rejections.py                          # Rejection rate analysis (publication figures)
python evaluation/gradcam_visualization.py                       # Grad-CAM attention maps
python evaluation/inference_benchmark.py                         # Latency/FPS/FLOP benchmarking
```

### Batch Training (Windows)

```bash
scripts/run_scut_baselines.bat          # All 9 models on SCUT
scripts/run_train_sca.bat               # SCA-MobileNet v2
scripts/run_cross_domain_eval.bat       # Cross-domain for all models
scripts/run_ablation_stn.bat            # STN ablation
scripts/run_ablation_ca.bat             # CoordAttention ablation
scripts/run_ablation_spp.bat            # SPP ablation
scripts/run_tongji_baselines.bat        # TONGJI dataset baselines
scripts/run_vera_baselines.bat          # VERA dataset baselines
scripts/run_gscl.bat                    # GSCL training
scripts/run_deit_swin.bat               # DeiT/Swin transformer experiments
```

### GSCL Subproject

**Requires separate conda environment** (Python 3.7.9, PyTorch 1.7.1, CUDA 11.0):

```bash
cd models/GSCL-PyTorch
conda env create --file env.yml
conda activate gscl
python vein_feature_learning/main_ssl.py   # Self-supervised pretraining
python vein_feature_learning/main_sl.py    # Supervised finetuning
```

## Dataset Format

Images organized as `data_dir/class_NNN/*.{png,jpg,bmp}`. Default input: 224×224 (FGFNet uses 256×256). Open-set protocol splits at the identity level (disjoint users in train/test). All datasets are in `datasets/`.

## Architecture Overview

### Entry Point: `train.py`

Central training hub that instantiates any model via `--model` flag, runs the training loop with mixed precision, performs biometric evaluation, and saves checkpoints. Also defines `SCATransformerBackbone` for timm-based backbone experiments.

### Model Architectures

| Model | Location | Key Design | Embedding |
|-------|----------|-----------|-----------|
| **RSNet** | `models/RSNet/model.py` | Dual-branch (local RLEB + global), MAB blocks, channel shuffle | 1024-d (local only at inference) |
| **SCA-MobileNet** | `models/SCA_MobileNet/model.py` | STN + MobileNetV3-Small-H (9 IR blocks) + SCIB (CoordAttention + SPP) | 1024-d |
| **FGFNet** | `models/FGFNet/model.py` | MobileViT + FFC (Fourier Feature Conv) + FFT spatial attention | varies |
| **VeinKAN** | `models/VeinKAN/model.py` | InceptionV3 (pretrained) + KAN classifier (3-layer, 540-dim, spline_order=3) | 2048-d |
| **EUSIPCO2020** | inline in `train.py` | Modified DenseNet-161 | varies |
| **MPSNet** | `models/MPSNet_2022/` | Multi-path spatial pyramid + SPP [1,2,4]; TF/Keras origin with PyTorch wrapper | varies |
| **GSCL** | `models/GSCL-PyTorch/` + `models/GSCL_2024/` | ResNet backbone, SimCLR/BYOL pretrain + StyleGAN2 synthetic data | 512/128-d |

### Shared Biometric Utilities (`models/biometric/`)

- `metrics.py`: `BiometricEvaluator` — genuine/imposter pair generation (balanced 1:1), cosine similarity, ROC analysis, EER, TAR@FAR, D-prime
- `data_augmentation.py`: `BiometricCompose` with vein-safe transforms (rotation ±10°, translation 10%, Gaussian noise, contrast/brightness, scale, perspective, gamma)
- `early_stopping.py`: `EarlyStopping` (patience=15, monitors val_eer) / `TrainingMonitor`
- `losses.py`: `ArcFaceLoss` (scale=30, margin=0.5, label_smoothing=0.1), `TripletLoss`, `CenterLoss`, `get_loss_function`
- `visualization.py`: `TrainingVisualizer` — 9 publication-quality figures (loss, EER, TAR@FAR, D-prime, score distributions, ROC, DET)
- `model_architectures.py`: `CBAM`, `EfficientNetBackbone`, shared attention modules

### Loss Functions

| Loss | Location | Details |
|------|----------|---------|
| **AdaFace** | `models/RSNet/losses.py` | Quality-adaptive margin (scale=50, margin=0.55, h=0.29) — RSNet default |
| **AdaCos** | `models/SCA_MobileNet/losses_adacos.py` | Dynamic scale, no fixed margin — SCA-MobileNet default |
| **FusionLoss** | `models/SCA_MobileNet/losses.py` | Triplet + Softmax (equal weights) |
| **FGFNetLoss** | `models/FGFNet/loss.py` | Contrastive between spatial and frequency branches (temp=0.1) |
| **DifferenceLoss** | `models/RSNet/losses.py` | Orthogonality constraint for RSNet dual-branch |
| **CosFace** | `models/GSCL_2024/loss/` | Cosine margin loss for GSCL |

### Database-Specific Configs

`--database` loads optimized loss hyperparams from `get_database_config()` / `DATABASE_CONFIGS` in `models/RSNet/losses.py`. Supported: `CASIA_850`, `CASIA_940`, `VERA`, `TJ_PV`, `PLUS_PV850`, `PLUS_PV950`, `SCUT`.

## Evaluation Protocol

Biometric verification (1:1 matching):
1. Extract embeddings from test set (dimension varies by model, typically 1024-d)
2. Generate genuine pairs (all intra-class) + imposter pairs (balanced 1:1 ratio)
3. Compute cosine similarity scores
4. Report: **EER**, **TAR@0.01%FAR**, **TAR@0.1%FAR**, **TAR@1%FAR**, **AUC**, **D-prime**

Best checkpoint saved as `results/<model>/checkpoints/best_model_eer.pth` (lowest EER). Per-epoch rejection analysis in `results/<model>/rejection_analysis/epoch_N_rejections.json`.

## Results Structure

```
results/
├── results_<model_name>/
│   ├── checkpoints/best_model_eer.pth
│   ├── training_metrics.json
│   └── rejection_analysis/epoch_*.json
├── results_cross_domain/
└── results_ablation_*/
```
