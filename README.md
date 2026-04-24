# SCA-MobileNet: Palm Vein Verification

Lightweight deep learning architecture for contactless palm vein biometric authentication under open-set protocol.

## Architecture

SCA-MobileNet integrates the **SCIB (Scale-Coordinate Invariant Block)** into a truncated MobileNetV3-Small backbone:

- **STN** (Spatial Transformer Network) — geometric normalization at input
- **Coordinate Attention** — spatial feature localization along both axes
- **Spatial Pyramid Pooling** — multi-scale representation (1×1, 2×2, 4×4)
- **AdaCos** loss with additive angular margin (m=0.35)

**3.19M parameters | 0.13G FLOPs | 10ms/image on GPU**

## Results

Open-set verification (identity-disjoint train/test split):

| Dataset | IDs | EER (%) | TAR@0.01% | AUC |
|---------|-----|---------|-----------|-----|
| Internal (NIR) | 1,549 | **0.89** | 96.93% | 0.9983 |
| TONGJI | 1,200 | **0.06** | 90.06% | 0.9999 |
| SCUT | 1,100 | **1.48** | 88.10% | 0.9986 |
| VERA | 220 | **2.76** | 86.20% | 0.9950 |

Ranked **#1 across all 7 scenarios** (4 in-domain + 3 cross-domain) against 11 baselines.

## Project Structure

```
├── train.py                 # Training entry point (all models)
├── models/
│   ├── SCA_MobileNet/       # Proposed model
│   ├── RSNet/               # RSNet + shared losses
│   ├── biometric/           # Metrics, augmentation, visualization
│   ├── FGFNet/              # FGFNet baseline
│   ├── VeinKAN/             # VeinKAN baseline
│   ├── GSCL_2024/           # GSCL wrapper
│   ├── MPSNet_2022/         # MPSNet baseline
│   └── Modified_Densenet161_2021/
├── preprocessing/           # Data preparation pipeline
├── evaluation/              # Analysis & evaluation scripts
├── scripts/                 # Batch training scripts
└── tools/                   # Utility scripts
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux
pip install -r requirements.txt
```

## Training

```bash
# SCA-MobileNet
python train.py --model sca_mobilenet --dataset datasets/SCUT_dataset_openset --database SCUT

# Baselines
python train.py --model rsnet --dataset datasets/TONGJI_dataset_openset
python train.py --model fgfnet --dataset datasets/VERA_dataset_openset

# Ablation (disable individual modules)
python train.py --model sca_mobilenet --dataset datasets/SCUT_dataset_openset --no-stn
python train.py --model sca_mobilenet --dataset datasets/SCUT_dataset_openset --no-ca
python train.py --model sca_mobilenet --dataset datasets/SCUT_dataset_openset --no-spp
```

Available models: `sca_mobilenet` | `rsnet` | `fgfnet` | `mpsnet` | `gscl` | `eusipco2020` | `mobilenetv3_uib`

## Evaluation

```bash
python evaluation/cross_domain_eval.py --model sca_mobilenet
python evaluation/inference_benchmark.py
python evaluation/statistical_significance.py
python evaluation/gradcam_visualization.py
```

## Data Preparation

```bash
# 1. Preprocessing (GrabCut → ROI → 128×128)
python preprocessing/palm_vein_preprocessing.py

# 2. CLAHE enhancement
python preprocessing/palm_vein_enhancement.py

# 3. Dataset-specific preparation
python preprocessing/prepare_scut_dataset.py

# 4. Open-set split (identity-level 70/30)
python preprocessing/split_scut_openset.py
```

Dataset format: `data_dir/class_NNN/*.{png,jpg,bmp}`

## Evaluation Protocol

1. Extract embeddings from test set (1024-d)
2. Generate genuine + impostor pairs (balanced 1:1)
3. Compute cosine similarity
4. Report: EER, TAR@FAR(0.01%, 0.1%, 1%), AUC, D-prime

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA (recommended)

## Citation

If you use this code, please cite:

```
Nguyễn Quốc Vương, Huỳnh Minh Quân, Bùi Danh Hường, Hoàng Văn Quý.
"SCA-MobileNet: Kiến trúc học sâu cho xác thực tĩnh mạch lòng bàn tay
không tiếp xúc với khối SCIB." 2025.
```
