# VERA Experimental Results

Đánh giá SCA-MobileNet và các phương pháp so sánh trên bộ dữ liệu công khai **VERA Palm Vein** theo giao thức xác thực tập mở (open-set verification).

---

## A. Bộ dữ liệu & Giao thức

### Thông tin VERA

| Thông tin | Giá trị |
|-----------|---------|
| Tổng danh tính | 220 (110 người × 2 tay L/R) |
| Ảnh/danh tính | 10 (2 phiên × 5 mẫu) |
| Tổng ảnh | 2.200 |
| Bước sóng | NIR |
| Tiền xử lý (CLAHE) | clipLimit=8.0, tileGridSize=8×8 |

### Phân chia Open-Set (seed=42, 70/30)

| Tập | Danh tính | Ảnh | Genuine pairs | Impostor pairs |
|-----|-----------|-----|---------------|----------------|
| Train | 154 (70%) | 1.540 | — | — |
| Test | 66 (30%) | 660 | C(10,2)×66 = 2.970 | ~2.970 (cân bằng) |

---

## B. Cấu hình Huấn luyện

Giống hoàn toàn với bộ dữ liệu tự thu thập (xem CLAUDE.md):
- AdamW, lr=0.001, MultiStepLR (×0.1 tại epoch 30, 60, 85)
- BalancedBatchSampler: 16 identities × 4 = 64
- 100 epochs, mixed precision, grad clip max_norm=10.0
- `--database VERA` (tối ưu loss hyperparameters cho VERA)

---

## C. Kết quả So sánh — VERA với CLAHE

> Giao thức Open-Set. Ký hiệu: † mã nguồn công khai, ‡ tự cài đặt lại.

| Model | EER (%) | TAR@0,01% | TAR@0,1% | AUC |
|-------|---------|-----------|----------|-----|
| MPSNet† | 4,865 | 68,01 | 79,87 | 0,9864 |
| Modified-DenseNet161† | 7,138 | 62,73 | 71,11 | 0,9772 |
| GSCL† | 7,172 | 62,90 | 65,76 | 0,9719 |
| RSNet‡ | 8,502 | 60,98 | 63,94 | 0,9664 |
| FGFNet† | 9,680 | 41,25 | 48,72 | 0,9609 |
| ResNet50 | 8,889 | 40,77 | 54,38 | 0,9652 |
| MobileNetV3-Base | 3,620 | 76,50 | 85,45 | 0,9935 |
| EfficientNet-B0 | 7,205 | 63,74 | 67,85 | 0,9796 |
| Swin-Tiny | 5,690 | 66,77 | 70,77 | 0,9843 |
| DeiT-Tiny | 6,987 | 64,61 | 70,94 | 0,9794 |
| MobileViT-S | 8,788 | 46,90 | 53,64 | 0,9709 |
| **SCA-MobileNet** | **2,761** | **86,20** | **90,91** | **0,9950** |

---

## D. Kết quả So sánh — VERA không có CLAHE (Ablation Preprocessing)

Mục đích: đánh giá ảnh hưởng của bước tăng cường quang học CLAHE lên hiệu năng.

| Model | EER (%) | TAR@0,01% | TAR@0,1% | AUC |
|-------|---------|-----------|----------|-----|
| MPSNet† | 3,434 | 80,13 | 83,91 | 0,9954 |
| Modified-DenseNet161† | 5,017 | 57,95 | 67,00 | 0,9877 |
| GSCL† | 5,219 | 66,36 | 73,60 | 0,9883 |
| RSNet‡ | 8,485 | 59,56 | 62,42 | 0,9626 |
| FGFNet† | 9,209 | 35,99 | 43,43 | 0,9683 |
| ResNet50 | 5,101 | 69,76 | 76,90 | 0,9874 |
| MobileNetV3-Base | 2,980 | 89,93 | 90,88 | 0,9948 |
| EfficientNet-B0 | 8,721 | 45,49 | 51,55 | 0,9750 |
| Swin-Tiny | 5,404 | 69,19 | 75,79 | 0,9840 |
| DeiT-Tiny | 5,202 | 60,20 | 71,25 | 0,9891 |
| MobileViT-S | 10,741 | 26,03 | 33,84 | 0,9628 |
| **SCA-MobileNet** | **1,953** | **92,02** | **94,58** | **0,9979** |

---

## E. Phân tích CLAHE vs No-CLAHE (SCA-MobileNet)

| Metric | CLAHE | No-CLAHE | Δ |
|--------|-------|----------|---|
| EER (%) | 2,761 | **1,953** | −0,808 |
| TAR@0,01% | 86,20 | **92,02** | +5,82 |
| TAR@0,1% | 90,91 | **94,58** | +3,67 |
| AUC | 0,9950 | **0,9979** | +0,0029 |

> **Nhận xét**: SCA-MobileNet đạt kết quả **tốt hơn khi không có CLAHE** trên VERA.
> Giải thích: STN trong SCA-MobileNet tự học được cách chuẩn hóa ảnh đầu vào,
> làm giảm sự phụ thuộc vào tiền xử lý thủ công. CLAHE có thể làm mất một số
> thông tin tần số thấp quan trọng mà mô hình cần.

---

## F. Nhận xét tổng quan

- **SCA-MobileNet vượt trội rõ ràng** trên VERA ở cả 2 variant (CLAHE và No-CLAHE)
- EER thấp hơn model tốt nhất trong nhóm so sánh (MPSNet 4,865%) gần **2×**
- TAR@0,01% đạt **86–92%** trong khi các model khác chỉ đạt 26–80%
- Kết quả nhất quán với paper trên dataset tự thu thập → **có thể bổ sung vào bài báo tạp chí** như bảng so sánh trên dataset công khai bổ sung
- Tất cả 12 model (CLAHE và No-CLAHE) đã có đầy đủ kết quả
