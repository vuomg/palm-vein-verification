# SCA-MobileNet: Palm Vein Verification

Lightweight deep learning model for contactless palm vein biometric authentication.  
Proposes the **SCIB block** (STN + CoordAttention + SPP) on a truncated MobileNetV3-Small backbone, evaluated under open-set identity-disjoint protocol across 4 datasets and 8 experimental scenarios.

> **Dataset:** HUTECH-CNPV1549 is available on Hugging Face with gated access — see [Dataset](#dataset) below.

---

## Architecture

```
Input (224×224)
  └─ STN                        # geometric alignment
      └─ MobileNetV3-Small-H    # 9 inverted-residual blocks, truncated
          └─ SCIB
              ├─ CoordAttention  # coordinate-wise spatial attention
              └─ SPP [1×1, 2×2, 4×4]  # multi-scale pooling
                  └─ Embedder → 1024-d feature vector
```

**3.19M parameters · 0.13 GFLOPs · ~10 ms/image (GPU)**

---

## Results

### In-domain verification (open-set, identity-disjoint 70/30 split)

| Dataset | Identities | EER (%) ↓ | TAR@0.01% ↑ | TAR@0.1% ↑ | AUC ↑ |
|---------|-----------|-----------|------------|-----------|-------|
| HUTECH-CNPV1549 | 1,549 | **0.89** | 96.93% | 98.49% | 0.9983 |
| TONGJI | 1,200 | **0.06** | 90.06% | 99.95% | 0.9999 |
| SCUT | 1,100 | **1.48** | 88.10% | 92.30% | 0.9986 |
| VERA | 220 | **2.76** | 86.20% | 90.90% | 0.9950 |

### Cross-domain generalization (no fine-tuning)

| Scenario | EER (%) ↓ | TAR@0.01% ↑ | TAR@0.1% ↑ | AUC ↑ |
|----------|-----------|------------|-----------|-------|
| HUTECH-CNPV1549 → TONGJI | **1.50** | 71.84% | 94.46% | 0.9981 |
| HUTECH-CNPV1549 → VERA | **3.60** | 89.25% | 90.53% | 0.9887 |
| SCUT → VERA | **5.61** | 78.07% | 82.28% | 0.9814 |
| VERA → SCUT | **10.13** | 47.53% | 56.94% | 0.9615 |

SCA-MobileNet ranks **#1 across all 8 scenarios** against 11 baseline models.

---

## Dataset

**HUTECH-CNPV1549** — 1,549 subjects, NIR palm vein, 10 images/subject (15,490 images total), captured at HUTECH University, Vietnam.

Access is restricted to non-commercial academic research. Submit a request at:

> 🤗 [huggingface.co/datasets/vuomg/HUTECH-CNPV1549](https://huggingface.co/datasets/vuomg/HUTECH-CNPV1549) *(coming soon)*

Requests require institutional affiliation and a description of research purpose. Approved within 3–5 business days.

Public benchmark datasets used in this work:
- [TONGJI](http://sse.tongji.edu.cn/jbhao/resource.html) — 600 subjects, contactless NIR
- [SCUT](https://www.scholat.com/team/biometricsveinlab) — 1,100 subjects, contactless NIR
- [VERA](https://www.idiap.ch/dataset/vera-palmvein) — 220 subjects, contactless NIR

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

If `ModuleNotFoundError: No module named 'biometric'`, add the project root to `PYTHONPATH`.

---

## Training

```bash
# SCA-MobileNet (proposed)
python train.py --model sca_mobilenet \
    --dataset datasets/SCUT_dataset_openset \
    --database SCUT --loss-type adacos_only \
    --feature-dim 1024 --batch-size 16 --epochs 100

# Baselines
python train.py --model rsnet   --dataset datasets/TONGJI_dataset_openset
python train.py --model fgfnet  --dataset datasets/VERA_dataset_openset
python train.py --model mpsnet  --dataset datasets/SCUT_dataset_openset

# Ablation — disable individual modules
python train.py --model sca_mobilenet --dataset datasets/SCUT_dataset_openset --no-stn
python train.py --model sca_mobilenet --dataset datasets/SCUT_dataset_openset --no-ca
python train.py --model sca_mobilenet --dataset datasets/SCUT_dataset_openset --no-spp
```

Available `--model` values: `sca_mobilenet` · `rsnet` · `fgfnet` · `mpsnet` · `gscl` · `eusipco2020`

Available `--sca-backbone` values: `mobilenetv3` (default) · `mobilevit_s` · `deit_tiny` · `swin_tiny` · `efficientnet_b0`

---

## Data Preparation

```bash
# 1. Preprocessing — GrabCut segmentation → ROI → 128×128
python preprocessing/palm_vein_preprocessing.py

# 2. CLAHE enhancement
python preprocessing/palm_vein_enhancement.py

# 3. Dataset-specific preparation
python preprocessing/prepare_tongji_dataset.py   # → TONGJI_enhanced/
python preprocessing/prepare_scut_dataset.py     # → SCUT_enhanced/
python preprocessing/prepare_vera_dataset.py     # → VERA_enhanced/

# 4. Open-set split (identity-level disjoint 70/30)
python preprocessing/split_tongji_openset.py     # → datasets/TONGJI_dataset_openset/
python preprocessing/split_scut_openset.py       # → datasets/SCUT_dataset_openset/
python preprocessing/split_vera_openset.py       # → datasets/VERA_dataset_openset/
```

Dataset format: `data_dir/identity_NNN/*.{png,jpg,bmp}`

---

## Evaluation

```bash
# Cross-domain generalization
python evaluation/cross_domain_eval.py --model sca_mobilenet --target tongji
python evaluation/cross_domain_eval.py --model sca_mobilenet --target vera

# Statistical significance (Wilcoxon, 5 seeds)
python evaluation/statistical_significance.py

# Grad-CAM attention maps
python evaluation/gradcam_visualization.py

# Latency / FLOPs benchmark
python evaluation/inference_benchmark.py

# Enrollment-probe protocol (5:5 split)
python evaluation/enrollment_probe_eval.py --model sca_mobilenet --dataset SCUT
```

---

## Evaluation Protocol

Biometric verification (1:1 matching):
1. Extract 1024-d embeddings from test identities
2. Generate genuine pairs (all intra-class) + impostor pairs (balanced 1:1)
3. Compute cosine similarity scores
4. Report: **EER**, **TAR@0.01% FAR**, **TAR@0.1% FAR**, **TAR@1% FAR**, **AUC**, **D-prime**

---

## Project Structure

```
SCA-MobileNet/
├── train.py                          # Training entry point
├── requirements.txt / INSTALL.md
├── models/
│   ├── SCA_MobileNet/                # Proposed model (STN + CA + SPP)
│   ├── RSNet/                        # RSNet baseline + AdaFace loss
│   ├── FGFNet/                       # FGFNet baseline
│   ├── VeinKAN/                      # VeinKAN baseline
│   ├── MPSNet_2022/                  # MPSNet baseline
│   ├── Modified_Densenet161_2021/    # Modified DenseNet-161
│   ├── GSCL_2024/                    # GSCL wrapper
│   └── biometric/                    # Shared: metrics, augmentation, losses
├── preprocessing/                    # Data preparation pipeline
└── evaluation/                       # Evaluation and analysis scripts
```

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{scamobilenet2025,
  author    = {Nguy{\~{e}}n, Qu{\'{o}}c V{\u{u}}{\sigma}ng and
               Hu{\`{y}}nh, Minh Qu{\^{a}}n and
               B{\`{u}}i, Danh H{\u{u}}{\sigma}ng and
               Ho{\`{a}}ng, V{\u{a}}n Qu{\'{y}}},
  title     = {{SCA-MobileNet}: A Lightweight Architecture for Contactless
               Palm Vein Verification with Scale-Coordinate Invariant Block},
  year      = {2025}
}
```

---

## License

This code is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — free for non-commercial academic research. The HUTECH-CNPV1549 dataset has a separate access agreement (see [Dataset](#dataset)).
