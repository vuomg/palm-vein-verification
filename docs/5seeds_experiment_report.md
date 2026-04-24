# 5-Seed Statistical Significance Experiment

**Mục tiêu**: Chạy 5 seeds × 5 models trên Internal Dataset, thu thập mean ± std, Wilcoxon test.  
**Bắt đầu**: 2026-04-23  
**Script**: `scripts/run_5seeds_significance.bat`  
**Output**: `results/results_5seeds/`

---

## Tiến độ tổng quan

- [ ] Chạy 25 runs (0/25 hoàn thành)
- [ ] Tổng hợp mean ± std
- [ ] Wilcoxon signed-rank test (SCA vs baselines)
- [ ] Cập nhật paper tables
- [ ] Review kết quả

---

## Chi tiết từng run

### SCA-MobileNet 3M (bottleneck, reduction=8)

| Seed | Status | Best EER (%) | Epoch | TAR@0.01%FAR | AUC | D-prime |
|------|--------|-------------|-------|--------------|-----|---------|
| 42 | ⬜ Pending | --- | --- | --- | --- | --- |
| 0 | ⬜ Pending | --- | --- | --- | --- | --- |
| 1 | ⬜ Pending | --- | --- | --- | --- | --- |
| 7 | ⬜ Pending | --- | --- | --- | --- | --- |
| 99 | ⬜ Pending | --- | --- | --- | --- | --- |

**Mean ± Std**: `---`

### SCA-MobileNet 12M (no bottleneck, reduction=32)

| Seed | Status | Best EER (%) | Epoch | TAR@0.01%FAR | AUC | D-prime |
|------|--------|-------------|-------|--------------|-----|---------|
| 42 | 🔄 Running (ep 20, 0.919%) | --- | --- | --- | --- | --- |
| 0 | ⬜ Pending | --- | --- | --- | --- | --- |
| 1 | ⬜ Pending | --- | --- | --- | --- | --- |
| 7 | ⬜ Pending | --- | --- | --- | --- | --- |
| 99 | ⬜ Pending | --- | --- | --- | --- | --- |

**Mean ± Std**: `---`

### RSNet

| Seed | Status | Best EER (%) | Epoch | TAR@0.01%FAR | AUC | D-prime |
|------|--------|-------------|-------|--------------|-----|---------|
| 42 | ⬜ Pending | --- | --- | --- | --- | --- |
| 0 | ⬜ Pending | --- | --- | --- | --- | --- |
| 1 | ⬜ Pending | --- | --- | --- | --- | --- |
| 7 | ⬜ Pending | --- | --- | --- | --- | --- |
| 99 | ⬜ Pending | --- | --- | --- | --- | --- |

**Mean ± Std**: `---`

### FGFNet

Config: feature-dim=640, batch-size=16, effective lr=1e-4

| Seed | Status | Best EER (%) | Epoch | TAR@0.01%FAR | AUC | D-prime |
|------|--------|-------------|-------|--------------|-----|---------|
| 42 | ⬜ Pending | --- | --- | --- | --- | --- |
| 0 | ⬜ Pending | --- | --- | --- | --- | --- |
| 1 | ⬜ Pending | --- | --- | --- | --- | --- |
| 7 | ⬜ Pending | --- | --- | --- | --- | --- |
| 99 | ⬜ Pending | --- | --- | --- | --- | --- |

**Mean ± Std**: `---`

### GSCL ResNet-18

Config: feature-dim=512, batch-size=64, lr=0.01

| Seed | Status | Best EER (%) | Epoch | TAR@0.01%FAR | AUC | D-prime |
|------|--------|-------------|-------|--------------|-----|---------|
| 42 | ⬜ Pending | --- | --- | --- | --- | --- |
| 0 | ⬜ Pending | --- | --- | --- | --- | --- |
| 1 | ⬜ Pending | --- | --- | --- | --- | --- |
| 7 | ⬜ Pending | --- | --- | --- | --- | --- |
| 99 | ⬜ Pending | --- | --- | --- | --- | --- |

**Mean ± Std**: `---`

---

## Tổng hợp kết quả

### Mean ± Std (EER %)

| Model | Params | EER (mean±std) | TAR@0.01%FAR | AUC | D-prime |
|-------|--------|---------------|--------------|-----|---------|
| **SCA-3M** | 3.19M | --- | --- | --- | --- |
| **SCA-12M** | 12.79M | --- | --- | --- | --- |
| RSNet | 21.6M | --- | --- | --- | --- |
| FGFNet | ~5.8M | --- | --- | --- | --- |
| GSCL ResNet-18 | 11.2M | --- | --- | --- | --- |

### Wilcoxon Signed-Rank Test (SCA-3M vs Baselines)

| Comparison | W-stat | p-value | Significant (p<0.05)? |
|-----------|--------|---------|----------------------|
| SCA-3M vs RSNet | --- | --- | --- |
| SCA-3M vs FGFNet | --- | --- | --- |
| SCA-3M vs GSCL | --- | --- | --- |
| SCA-3M vs SCA-12M | --- | --- | --- |

### Wilcoxon Signed-Rank Test (SCA-12M vs Baselines)

| Comparison | W-stat | p-value | Significant (p<0.05)? |
|-----------|--------|---------|----------------------|
| SCA-12M vs RSNet | --- | --- | --- |
| SCA-12M vs FGFNet | --- | --- | --- |
| SCA-12M vs GSCL | --- | --- | --- |

---

## Cập nhật Paper

- [ ] Table: Internal dataset SOTA → thêm cột mean ± std
- [ ] Table: Statistical significance → Wilcoxon results
- [ ] Discussion: phân tích robustness across seeds
- [ ] Kết luận: cập nhật claim dựa trên evidence

---

## Ghi chú

- **Config chung**: epochs=100, eval-frequency=5, MultiStepLR [30,60,85] gamma=0.1
- **Dataset**: `datasets/final_dataset_openset` (1084 train IDs, 463 test IDs)
- **Môi trường**: PyTorch 2.5.1+cu121, RTX 4050 Laptop GPU (6GB)
- **12M hiện tại**: Đang chạy riêng tại `results/results_sca_12m_r32_sca_mobilenet/` (seed 42), sẽ copy kết quả vào 5seeds folder khi xong
- **Old checkpoints** (không ghi đè): `results/results_sca_sca_mobilenet/`, `results/results_sca_v2_sca_mobilenet/`
