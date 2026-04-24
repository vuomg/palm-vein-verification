# Installation Guide - Palm Vein Biometric Authentication

## Requirements
- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)
- Linux/Windows/Mac

## Quick Start

### Linux Setup

**Option 1: Automatic Setup (Recommended)**
```bash
chmod +x setup.sh
./setup.sh
```

**Option 2: Manual Setup**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt

# (Optional) For GPU support instead of CPU:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Windows Setup

**Option 1: Automatic Setup**
```cmd
setup.bat
```

**Option 2: Manual Setup**
```cmd
# Create virtual environment
python -m venv venv
venv\Scripts\activate.bat

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Verify Installation

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
python3 -c "import timm; print('TIMM: OK')"
```

---

## Full Pipeline: SCUT Dataset

### 1. Prepare Dataset
```bash
python prepare_scut_dataset.py
```
Output: `SCUT_enhanced/` (1100 classes, ~11k images with CLAHE enhancement)

### 2. Split Dataset (70% train / 30% test)
```bash
python split_scut_openset.py
```
Output: `SCUT_dataset_openset/` (train/ + test/ with identity-level split)

### 3. Train All 9 Models

**Linux:**
```bash
chmod +x run_scut_baselines.sh
./run_scut_baselines.sh
```

**Windows:**
```cmd
run_scut_baselines.bat
```

**Models trained:**
1. MPSNet (Multi-Path Spatial Pyramid)
2. Modified-DenseNet161 (EUSIPCO 2020)
3. GSCL (ResNet18)
4. RSNet (Dual-branch RLEB)
5. FGFNet (MobileViT + Fourier)
6. ResNet50 (GSCL backbone)
7. MobileNetV3-Base (baseline)
8. EfficientNet-B0 (efficient backbone)
9. SCA-MobileNet (proposed)

### 4. Results
Results saved in: `results_scut_*/`
- `checkpoints/best_model_eer.pth` - best model checkpoint
- `training_metrics.json` - EER, TAR@FAR, AUC, D-prime

---

## Train Single Model

```bash
python train.py \
    --model sca_mobilenet \
    --dataset SCUT_dataset_openset \
    --database SCUT \
    --batch-size 16 \
    --epochs 100 \
    --lr 0.001 \
    --eval-frequency 5
```

**Available models:**
- `rsnet` - RSNet
- `sca_mobilenet` - SCA-MobileNet (default)
- `fgfnet` - FGFNet
- `mpsnet` - MPSNet
- `eusipco2020` - Modified-DenseNet161
- `gscl` - GSCL

**Available databases (--database):**
- `SCUT` - SCUT dataset
- `VERA` - VERA dataset
- `CASIA_850`, `CASIA_940` - CASIA datasets
- `TJ_PV` - TONGJI dataset
- `PLUS_PV850`, `PLUS_PV950` - PLUSVein datasets

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'biometric'"
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```
Or add to script header.

### PyTorch installation issues
```bash
# For CPU-only
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory
Reduce batch size:
```bash
python train.py --model sca_mobilenet --dataset SCUT_dataset_openset --batch-size 8
```

### GPU not detected
```bash
python3 -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

---

## Dependencies

See `requirements.txt` for full list:
- **PyTorch 2.0+** - Deep learning framework
- **OpenCV 4.8+** - Computer vision
- **NumPy 1.24+** - Numerical computing
- **Pillow 10.0+** - Image processing
- **Scikit-learn 1.3+** - Machine learning utilities
- **Matplotlib 3.8+** - Visualization
- **TIMM 0.9+** - Pre-trained models
- **TensorBoard 2.14+** - Training metrics logging

---

## Project Structure

```
srcPalmVein/
├── train.py                     # Main training script
├── prepare_scut_dataset.py      # SCUT dataset preparation
├── split_scut_openset.py        # Open-set split
├── run_scut_baselines.bat       # Windows training script
├── run_scut_baselines.sh        # Linux training script
├── requirements.txt             # Python dependencies
├── setup.bat                    # Windows setup
├── setup.sh                     # Linux setup
├── biometric/                   # Shared utilities
│   ├── metrics.py              # BiometricEvaluator
│   ├── data_augmentation.py    # Augmentation pipeline
│   ├── early_stopping.py       # EarlyStopping
│   ├── losses.py               # Loss functions
│   └── visualization.py        # Visualization utils
├── core/                        # RSNet model
├── SCA_MobileNet/              # SCA-MobileNet model
├── FGFNet/                     # FGFNet model
├── GSCL-PyTorch/              # GSCL subproject
└── results_scut_*/             # Training results
```

---

## Citations & References

- **RSNet**: Dual-branch architecture with MAB and channel shuffle
- **SCA-MobileNet**: Spatial-Coordinate Attention + MobileNetV3 + SPP
- **FGFNet**: MobileViT + Fourier Feature Convolution + FFT attention
- **GSCL**: Generative Self-Supervised Contrastive Learning

---

## Support

For issues, check:
1. `CLAUDE.md` - Project overview & common commands
2. `requirements.txt` - Dependency versions
3. `train.py --help` - Training arguments
