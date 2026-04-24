# Plan: Cross-Domain VERA ↔ SCUT & Đề xuất mô hình

## Yêu cầu từ thầy

1. **Cross-domain VERA vs SCUT**: Train trên VERA → test trên SCUT
2. **Cross-domain SCUT vs VERA**: Train trên SCUT → test trên VERA
3. **Đề xuất mô hình riêng**: Thầy sẽ đưa thêm tài liệu để đề xuất kiến trúc mới

---

## Phần 1: Cross-Domain VERA ↔ SCUT

### Mục tiêu

Đánh giá khả năng **generalization** của các model khi train trên dataset này, test trên dataset khác (domain shift):
- **VERA**: 220 identities, finger vein, 154 train / 66 test
- **SCUT**: 1100 identities, palm vein, 770 train / 330 test

Cross-domain dùng **toàn bộ** target dataset (train + test gộp lại) vì model chưa bao giờ thấy data từ target domain.

### Checkpoint — ĐẦY ĐỦ 12/12 model cả 2 chiều

| # | Model | VERA `.pth` | SCUT `.pth` | VERA→SCUT | SCUT→VERA |
|---|-------|:-:|:-:|:-:|:-:|
| 1 | SCA-MobileNet (proposed) | ✅ | ✅ | ✅ | ✅ |
| 2 | MPSNet | ✅ | ✅ | ✅ | ✅ |
| 3 | Modified-DenseNet161 | ✅ | ✅ | ✅ | ✅ |
| 4 | RSNet | ✅ | ✅ | ✅ | ✅ |
| 5 | FGFNet | ✅ | ✅ | ✅ | ✅ |
| 6 | MobileNetV3-Base | ✅ | ✅ | ✅ | ✅ |
| 7 | EfficientNet-B0 | ✅ | ✅ | ✅ | ✅ |
| 8 | DeiT-Tiny | ✅ | ✅ | ✅ | ✅ |
| 9 | Swin-Tiny | ✅ | ✅ | ✅ | ✅ |
| 10 | MobileViT-S | ✅ | ✅ | ✅ | ✅ |
| 11 | GSCL (ResNet18) | ✅ | ✅ | ✅ | ✅ |
| 12 | ResNet50 | ✅ | ✅ | ✅ | ✅ |

**Ghi chú**: GSCL + ResNet50 checkpoint nằm trong `models/GSCL-PyTorch/vein_feature_learning/results/` (train bằng framework riêng, nhưng cùng kiến trúc ResNets — load được bình thường).

### Script đã tạo

| File | Mô tả |
|------|-------|
| `evaluation/cross_domain_vera_scut.py` | Script cross-domain, hỗ trợ `--direction vera_to_scut / scut_to_vera` |
| `scripts/run_cross_domain_vera_scut.bat` | Batch chạy tất cả 12 models cả 2 chiều |

### Cách chạy

```bash
# Chạy tất cả (12 model x 2 chiều = 24 evaluations)
scripts\run_cross_domain_vera_scut.bat

# Từng direction
python evaluation/cross_domain_vera_scut.py --direction vera_to_scut --all
python evaluation/cross_domain_vera_scut.py --direction scut_to_vera --all

# Từng model
python evaluation/cross_domain_vera_scut.py --direction vera_to_scut --model sca_mobilenet
python evaluation/cross_domain_vera_scut.py --direction scut_to_vera --model sca_mobilenet

# Kết quả lưu tại
# results/results_cross_domain_vera_scut/cross_domain_vera_to_scut_<model>.json
# results/results_cross_domain_vera_scut/cross_domain_scut_to_vera_<model>.json
# results/results_cross_domain_vera_scut/summary_vera_to_scut.json
# results/results_cross_domain_vera_scut/summary_scut_to_vera.json
```

### Output format

Mỗi model tạo 1 file JSON với metrics: **EER**, **TAR@0.01%FAR**, **TAR@0.1%FAR**, **TAR@1%FAR**, **AUC**, **D-prime**.

Cuối cùng có summary table sorted theo EER (model nào generalize tốt nhất).

---

## Phần 2: Đề xuất mô hình riêng (chờ thầy)

Thầy sẽ đưa thêm tài liệu/ý tưởng. Kết quả cross-domain sẽ là **evidence** để:

- Xác định model nào generalize tốt/kém giữa VERA (finger vein) và SCUT (palm vein)
- Phân tích **tại sao** SCA-MobileNet generalize tốt hơn (hoặc kém hơn) so với baselines
- Justify đề xuất cải tiến kiến trúc dựa trên bottleneck quan sát được

### Các hướng đề xuất tiềm năng

1. **Domain adaptation module**: Thêm feature alignment layer giữa source/target domain
2. **Attention-based adaptation**: Cải tiến CoordAttention để tự adapt theo domain
3. **Multi-scale feature fusion**: Kết hợp features ở nhiều resolution khác nhau
4. **Contrastive cross-domain learning**: Dùng contrastive loss để học domain-invariant features

*(Chờ tài liệu từ thầy để xác định hướng cụ thể)*

---

## Timeline đề xuất

| Bước | Công việc | Thời gian |
|------|-----------|-----------|
| 1 | Chạy cross-domain VERA→SCUT (12 models) | ~3-4 giờ GPU |
| 2 | Chạy cross-domain SCUT→VERA (12 models) | ~2-3 giờ GPU |
| 3 | Phân tích kết quả, tạo bảng so sánh | 1 giờ |
| 4 | Chờ thầy đưa tài liệu → đề xuất mô hình | TBD |
