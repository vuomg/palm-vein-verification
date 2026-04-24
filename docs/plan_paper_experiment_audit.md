# Kiểm tra hoàn chỉnh thực nghiệm cho Paper Hội nghị

## Tổng quan Paper

Paper journal version (ICT Dong Thap 2026) so sánh **SCA-MobileNet** với 11 baseline trên 4 dataset + cross-domain + ablation.

---

## 1. Bảng tổng hợp: Paper yêu cầu vs Kết quả đã có

### Ký hiệu
- ✅ = Đã train + có metrics + có `.pth`
- 📊 = Có metrics nhưng không có `.pth`

- ❌ = Chưa train / chưa có kết quả
- 📝 = Đã có trong paper

---

### Table I: Dataset Summary — ✅ OK
- Internal (1,549 ID), TONGJI (1,200 ID), SCUT (1,100 ID), VERA (220 ID)
- Thông tin đầy đủ trong paper

---


### Table II: Ablation Study (Internal dataset) — ✅ OK
8 config (toggle STN/CA/SPP), metrics: EER, TAR@0.01%, AUC, Params, FLOPs

| Config | Paper | Kết quả |
|--------|:---:|:---:|
| 1. Base MobileNetV3 | 📝 | ✅ |
| 2. +STN | 📝 | ✅ |
| 3. +CA | 📝 | ✅ |
| 4. +SPP | 📝 | ✅ |
| 5. STN+CA | 📝 | ✅ |
| 6. STN+SPP | 📝 | ✅ |
| 7. CA+SPP | 📝 | ✅ |
| 8. SCA-MobileNet (full) | 📝 | ✅ |

**Vấn đề**: EER Config 8 = 0.89% (conference) vs 0.90% (journal ablation) vs 0.89% (journal comparison). **Cần thống nhất**.

---

### Table III: Internal Dataset Comparison (9 models) — ⚠️ THIẾU Transformers

| Model | Paper (9) | Kết quả | Ghi chú |
|-------|:---:|:---:|---------|
| MPSNet | 📝 | ✅ | |
| Modified-DenseNet161 | 📝 | ✅ | |
| GSCL (ResNet18) | 📝 | ✅ | |
| RSNet | 📝 | ✅ | |
| FGFNet | 📝 | ✅ | |
| ResNet50 | 📝 | ✅ | |
| MobileNetV3-Base | 📝 | ✅ | |
| EfficientNet-B0 | 📝 | ✅ | |
| SCA-MobileNet | 📝 | ✅ | |
| Swin-Tiny | — | ❌ | SCUT/VERA có nhưng Internal chưa train |
| DeiT-Tiny | — | ❌ | Tương tự |
| MobileViT-S | — | ❌ | Tương tự |

**Nhận xét**: Paper hiện chỉ có 9 model trên Internal. SCUT/VERA có 12 model. Nếu muốn thống nhất 12 model trên tất cả dataset → cần train 3 transformer trên Internal.

---

### Table IV: TONGJI Results (6 models) — ⚠️ THIẾU 6 models

| Model | Paper (6) | Kết quả | Ghi chú |
|-------|:---:|:---:|---------|
| MPSNet | 📝 | ✅ `.pth` có, metrics thiếu JSON | |
| Modified-DenseNet161 | 📝 | ✅ `.pth` có, metrics thiếu JSON | |
| GSCL (ResNet18) | 📝 | ✅ `.pth` có, metrics thiếu JSON | |
| RSNet | 📝 | ✅ `.pth` có, metrics thiếu JSON | |
| FGFNet | 📝 | ✅ `.pth` có, metrics thiếu JSON | |
| SCA-MobileNet | 📝 | ✅ `.pth` có, metrics thiếu JSON | |
| ResNet50 | — | ❌ | Chưa train trên TONGJI |
| MobileNetV3-Base | — | ❌ | Chưa train |
| EfficientNet-B0 | — | ❌ | Chưa train |
| Swin-Tiny | — | ❌ | Chưa train |
| DeiT-Tiny | — | ❌ | Chưa train |
| MobileViT-S | — | ❌ | Chưa train |

**Vấn đề lớn**:
1. Paper chỉ có **6 model** trên TONGJI, trong khi SCUT/VERA có **12 model**
2. Tất cả 7 TONGJI results **thiếu `training_metrics.json`** (train bằng pipeline cũ)
3. Muốn thống nhất → cần train thêm **6 model** trên TONGJI

---

### Table V: SCUT Results (12 models) — ✅ ĐẦY ĐỦ

| Model | Paper (12) | Kết quả |
|-------|:---:|:---:|
| MPSNet | 📝 | ✅ |
| Modified-DenseNet161 | 📝 | ✅ |
| GSCL (ResNet18) | 📝 | 📊 (metrics có, `.pth` ở GSCL-PyTorch) |
| RSNet | 📝 | ✅ |
| FGFNet | 📝 | ✅ |
| ResNet50 | 📝 | 📊 (metrics có, `.pth` ở GSCL-PyTorch) |
| MobileNetV3-Base | 📝 | ✅ |
| EfficientNet-B0 | 📝 | ✅ |
| Swin-Tiny | 📝 | ✅ |
| DeiT-Tiny | 📝 | ✅ |
| MobileViT-S | 📝 | ✅ |
| SCA-MobileNet | 📝 | ✅ |

---

### Table VI: VERA with CLAHE (12 models) — ✅ ĐẦY ĐỦ

| Model | Paper (12) | Kết quả |
|-------|:---:|:---:|
| Tất cả 12 model | 📝 | ✅ hoặc 📊 (GSCL/ResNet50) |

---

### Table VII: VERA without CLAHE (12 models) — ✅ ĐẦY ĐỦ

| Model | Paper (12) | Kết quả |
|-------|:---:|:---:|
| Tất cả 12 model | 📝 | ✅ hoặc 📊 (GSCL/ResNet50) |

---

### Table VIII: CLAHE Ablation (5 models) — ✅ OK
So sánh EER with/without CLAHE cho 5 model top. Dữ liệu lấy từ Table VI + VII.

---

### Table IX: Cross-Domain NIR→TONGJI (6 models) — ⚠️ CÓ THỂ MỞ RỘNG

| Model | Paper (6) | Kết quả |
|-------|:---:|:---:|
| MPSNet | 📝 | ✅ |
| Modified-DenseNet161 | 📝 | ✅ |
| GSCL | 📝 | ✅ |
| RSNet | 📝 | ✅ |
| FGFNet | 📝 | ✅ |
| SCA-MobileNet | 📝 | ✅ |

**Nhận xét**: Chỉ 6 model. Có thể mở rộng lên 12 nhưng cần train thêm 6 model trên Internal (đã có `.pth`).

---

### Cross-Domain VERA↔SCUT — ❌ CHƯA CÓ TRONG PAPER, CHƯA CHẠY

Đây là yêu cầu mới từ thầy:
- Script đã tạo: `evaluation/cross_domain_vera_scut.py`
- 12 model x 2 chiều = 24 evaluations
- **CHƯA chạy**

---

### Table X: Summary SCA-MobileNet (6 rows) — ✅ OK
Tổng hợp từ các bảng trên.

---

## 2. Tổng hợp vấn đề cần giải quyết

### Mức ưu tiên CAO (cần cho paper hoàn chỉnh)

| # | Vấn đề | Chi tiết | Ước tính thời gian |
|---|--------|----------|-------------------|
| 1 | **TONGJI thiếu 6 model** | ResNet50, MobileNetV3-Base, EfficientNet-B0, Swin-Tiny, DeiT-Tiny, MobileViT-S chưa train trên TONGJI | ~12-15h GPU |
| 2 | **TONGJI thiếu metrics JSON** | 7 kết quả cũ không có `training_metrics.json` | Chạy lại hoặc extract từ log |
| 3 | **Cross-domain VERA↔SCUT chưa chạy** | Script đã tạo, cần chạy 24 evaluations | ~5-6h GPU |
| 4 | **EER inconsistency (0.89% vs 0.90%)** | Không phải bug metric — 2 lần train khác nhau. Ablation Config 8 = 0.90%, Comparison = 0.89%. **Cần chọn 1 giá trị thống nhất** | Sửa LaTeX |

### Mức ưu tiên TRUNG BÌNH (nâng cao chất lượng)

| # | Vấn đề | Chi tiết | Ước tính |
|---|--------|----------|----------|
| 6 | **Internal thiếu 3 transformer** | Swin-Tiny, DeiT-Tiny, MobileViT-S chưa train trên Internal dataset | ~6-8h GPU |
| 7 | **Cross-domain mở rộng** | NIR→TONGJI chỉ 6 model, có thể lên 12 | ~3h GPU |

### Mức ưu tiên THẤP (cleanup)

| # | Vấn đề | Chi tiết |
|---|--------|----------|
| 9 | Dọn kết quả cũ/trùng | `result_scut/` (bản cũ), `results_eusipco2020` (rỗng), `results_resnet50_gscl` (incomplete) |
| 10 | Legacy results thiếu metrics | 17 thư mục kết quả cũ (no dataset prefix) không có JSON |

---

## 3. Plan thực hiện

### Phase 1: Train TONGJI thiếu (ưu tiên cao nhất)

Tạo `scripts/run_tongji_full.bat` để train 6 model còn thiếu + re-train 6 model cũ (để có metrics JSON):

```
TONGJI cần train:
1. ResNet50 (GSCL framework)       — ~2h
2. MobileNetV3-Base (--no-stn --no-ca --no-spp) — ~1.5h
3. EfficientNet-B0                  — ~1.5h
4. Swin-Tiny                        — ~2h
5. DeiT-Tiny                        — ~2h
6. MobileViT-S                      — ~2h
```

Tổng: ~11h GPU

### Phase 2: Chạy Cross-Domain VERA↔SCUT

```bash
scripts\run_cross_domain_vera_scut.bat
```

12 model x 2 chiều = 24 evaluations, ~5h GPU.

### Phase 3: Sửa paper

1. Thống nhất EER (0.89% hay 0.90%)
2. Thêm bảng cross-domain VERA↔SCUT
3. Cân nhắc thêm 3 transformer vào bảng Internal + TONGJI

### Phase 4 (tùy chọn): Bổ sung Internal transformer

```
Internal cần train (nếu muốn 12 model):
1. Swin-Tiny    — ~2h
2. DeiT-Tiny    — ~2h  
3. MobileViT-S  — ~2h
```

---

## 4. Ma trận hoàn chỉnh: Model × Dataset

| Model | Internal | TONGJI | SCUT | VERA-CLAHE | VERA-NoCLAHE | Cross NIR→TJ | Cross VERA↔SCUT |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SCA-MobileNet | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | script ready |
| MPSNet | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | script ready |
| DenseNet161 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | script ready |
| GSCL (ResNet18) | ✅ | ✅ | 📊 | 📊 | 📊 | ✅ | script ready |
| RSNet | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | script ready |
| FGFNet | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | script ready |
| ResNet50 | ✅ | ❌ | 📊 | 📊 | 📊 | — | script ready |
| MobileNetV3-Base | ✅ | ❌ | ✅ | ✅ | ✅ | — | script ready |
| EfficientNet-B0 | ✅ | ❌ | ✅ | ✅ | ✅ | — | script ready |
| Swin-Tiny | ❌ | ❌ | ✅ | ✅ | ✅ | — | script ready |
| DeiT-Tiny | ❌ | ❌ | ✅ | ✅ | ✅ | — | script ready |
| MobileViT-S | ❌ | ❌ | ✅ | ✅ | ✅ | — | script ready |

**Ký hiệu**: ✅ = đầy đủ (.pth + metrics) | 📊 = metrics có, .pth ở GSCL-PyTorch | ❌ = chưa train

---

## 5. Khuyến nghị cho paper hội nghị

1. **Bắt buộc**: Chạy cross-domain VERA↔SCUT (yêu cầu thầy)
2. **Bắt buộc**: Thống nhất EER trong paper — chọn 1 giá trị (0.89% hoặc 0.90%) cho cả ablation + comparison
3. **Nên làm**: Train TONGJI đầy đủ 12 model (để bảng thống nhất)
4. **Tùy chọn**: Train Internal + transformer baselines
