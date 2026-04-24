# Plan: Bổ sung Hình ảnh & Sơ đồ cho Paper

## Hiện trạng

### 10 hình đang dùng trong paper:
| # | File | Mô tả | Section |
|---|------|-------|---------|
| 1 | fig_pose_variation.png | Biến thiên tư thế/khoảng cách | III - Dataset |
| 2 | fig_framework.png | Khung tổng thể open-set | IV - Method |
| 3 | fig_preprocessing.png | Pipeline tiền xử lý 3 giai đoạn | IV - Preprocessing |
| 4 | fig_segmentation.png | GrabCut segmentation | IV - Preprocessing |
| 5 | fig_scale_norm.png | Chuẩn hóa tỷ lệ Distance Transform | IV - Preprocessing |
| 6 | fig_clahe.png | CLAHE enhancement | IV - Preprocessing |
| 7 | fig_architecture.png | Kiến trúc SCA-MobileNet + SCIB | IV - Architecture |
| 8 | fig_evaluation_pipeline.png | Pipeline đánh giá verification | V - Protocol |
| 9 | fig_stn_analysis.png | Phân tích biến đổi STN | V - Ablation |
| 10 | fig_roc.png | Đường cong ROC | V - Performance |

### 4 hình có sẵn nhưng CHƯA dùng:
| File | Mô tả | Nên thêm? |
|------|-------|-----------|
| fig_det.png | Đường cong DET | ✅ Rất cần (biometric standard) |
| fig_det_curve_advanced.png | DET nâng cao (2 thang đo) | ✅ Thay thế fig_det nếu đẹp hơn |
| fig_tsne.png | t-SNE embedding visualization | ✅ Rất cần (minh họa cluster) |
| fig_roc_curve_epoch50.png | ROC tại epoch 50 | ❌ Không cần (đã có ROC cuối) |

### 4 drawio templates (cần export PNG):
| File | Mô tả |
|------|-------|
| docs/fig_architecture.drawio | Kiến trúc (có thể cập nhật) |
| docs/preprocessing_pipeline_diagram.drawio | Pipeline preprocessing |
| docs/evaluation_protocol_diagram.drawio | Evaluation protocol |
| docs/pipeline_diagram.drawio | Pipeline tổng thể |

---

## Đề xuất bổ sung (ưu tiên cao → thấp)

### 🔴 Ưu tiên cao (nên có trong journal)

#### 1. DET Curve (fig_det.png) — CÓ SẴN
- **Lý do**: Chuẩn ISO 19795 yêu cầu DET curve cho biometric paper
- **Vị trí**: Cạnh ROC curve (Section V - Performance Analysis)
- **Việc cần làm**: Thêm `\includegraphics` vào paper, viết caption
- **Thời gian**: 5 phút

#### 2. t-SNE Embedding Visualization (fig_tsne.png) — CÓ SẴN
- **Lý do**: Minh họa trực quan chất lượng không gian embedding (cluster separation)
- **Vị trí**: Section V - sau SOTA comparison hoặc Discussion
- **Việc cần làm**: Thêm vào paper + viết phân tích
- **Thời gian**: 5 phút

#### 3. Score Distribution Plot — CẦN TẠO MỚI
- **Mô tả**: Histogram/KDE plot phân bố genuine vs impostor scores
- **Lý do**: Minh họa rõ d-prime, overlap giữa 2 phân bố → giải thích EER
- **Script**: `evaluation/gradcam_visualization.py` hoặc tạo mới từ training_metrics
- **Vị trí**: Section V - Performance Analysis, cạnh ROC/DET
- **Thời gian**: ~30 phút

#### 4. Cross-domain EER Comparison Bar Chart — CẦN TẠO MỚI
- **Mô tả**: Grouped bar chart: 12 models × 3 kịch bản cross-domain
  - X-axis: models (sorted by avg EER)
  - Y-axis: EER (%)
  - 3 bars per model: NIR→TONGJI, VERA→SCUT, SCUT→VERA
- **Lý do**: So sánh trực quan hiệu năng cross-domain — SCA-MobileNet luôn thấp nhất
- **Vị trí**: Section V - Cross-domain evaluation
- **Thời gian**: ~1h (matplotlib script)

#### 5. Dataset Sample Images — CẦN TẠO MỚI
- **Mô tả**: Grid 4×3 hoặc 4×2 hiển thị sample images từ 4 datasets
  - Row 1: Internal (NIR palm vein) — raw + ROI + CLAHE
  - Row 2: TONGJI (visible palmprint) — ROI + CLAHE
  - Row 3: SCUT (visible palmprint) — ROI + CLAHE
  - Row 4: VERA (NIR finger vein) — ROI + CLAHE
- **Lý do**: Reviewer cần thấy data thực tế, sự khác biệt giữa datasets
- **Vị trí**: Section III - Datasets
- **Thời gian**: ~1h (chọn sample + compose grid)

### 🟡 Ưu tiên trung bình

#### 6. Grad-CAM Attention Maps — CẦN TẠO MỚI
- **Mô tả**: Grad-CAM heatmap trên ảnh test, so sánh:
  - SCA-MobileNet (STN+CA+SPP) vs MobileNetV3 Base
  - Highlight vùng mạch máu mà model chú ý
- **Lý do**: Chứng minh CA thực sự focus vào vein structure, không phải background
- **Script**: `evaluation/gradcam_visualization.py` (đã có)
- **Vị trí**: Section V - Ablation hoặc Discussion
- **Thời gian**: ~1h (chạy script + chọn samples đẹp)

#### 7. Training Convergence Curves — CẦN TẠO MỚI
- **Mô tả**: EER vs Epoch cho top-4 models trên 1 dataset (e.g., TONGJI)
  - Lines: SCA-MobileNet, RSNet, MPSNet, MobileNetV3 Base
  - Highlight epoch tốt nhất
- **Lý do**: Cho thấy tốc độ hội tụ, ổn định training
- **Dữ liệu**: training_metrics.json (đã có)
- **Vị trí**: Section V - Results
- **Thời gian**: ~30 phút

#### 8. Ablation Bar Chart — CẦN TẠO MỚI
- **Mô tả**: Bar chart 8 configs từ ablation study
  - X-axis: Config 1-8
  - Y-axis: EER (%) + TAR@0.01% (dual axis)
  - Color-coded theo components (STN/CA/SPP)
- **Lý do**: Trực quan hơn bảng số, thấy rõ đóng góp từng module
- **Vị trí**: Section V - Ablation Study (cạnh bảng hiện tại)
- **Thời gian**: ~30 phút

#### 9. Model Efficiency Scatter Plot — CẦN TẠO MỚI
- **Mô tả**: Scatter plot: X = Params (M), Y = EER (%)
  - Mỗi điểm = 1 model, kích thước = FLOPs
  - Highlight Pareto frontier
  - SCA-MobileNet nổi bật: params vừa phải, EER thấp nhất
- **Lý do**: Minh chứng trực quan efficiency vs accuracy trade-off
- **Vị trí**: Section V - Discussion hoặc Comparison
- **Thời gian**: ~30 phút

### 🟢 Tùy chọn (nice-to-have)

#### 10. Data Augmentation Examples — CẦN TẠO MỚI
- **Mô tả**: Grid hiển thị original + 6 augmented versions
  - Rotation, translation, scale, Gaussian noise, contrast, brightness
- **Lý do**: Minh họa augmentation strategy
- **Vị trí**: Section IV - Preprocessing
- **Thời gian**: ~30 phút

#### 11. Cross-domain EER Heatmap — CẦN TẠO MỚI
- **Mô tả**: Confusion matrix style heatmap
  - Rows = Source dataset, Cols = Target dataset
  - Cell = EER (%) of SCA-MobileNet
  - Color: green (low EER) → red (high EER)
- **Lý do**: Tổng quan nhanh cross-domain performance
- **Vị trí**: Section V - Cross-domain summary
- **Thời gian**: ~30 phút

#### 12. SCIB Block Diagram (chi tiết) — CẦN TẠO MỚI
- **Mô tả**: Sơ đồ chi tiết luồng tensor qua SCIB block
  - Input tensor shape → CA → shape → SPP branches → concat → embed
  - Ghi rõ kích thước tensor tại mỗi bước
- **Lý do**: fig_architecture.png hiện tại khá tổng quát, cần zoom vào SCIB
- **Tool**: draw.io hoặc tikz
- **Vị trí**: Section IV - SCIB Design
- **Thời gian**: ~2h

---

## Tổng hợp ưu tiên

| # | Hình | Trạng thái | Thời gian | Ưu tiên |
|---|------|-----------|-----------|---------|
| 1 | DET Curve | ✅ Có sẵn | 5 phút | 🔴 |
| 2 | t-SNE Embedding | ✅ Có sẵn | 5 phút | 🔴 |
| 3 | Score Distribution | ❌ Cần tạo | 30 phút | 🔴 |
| 4 | Cross-domain Bar Chart | ❌ Cần tạo | 1h | 🔴 |
| 5 | Dataset Samples | ❌ Cần tạo | 1h | 🔴 |
| 6 | Grad-CAM Maps | ❌ Cần tạo | 1h | 🟡 |
| 7 | Training Convergence | ❌ Cần tạo | 30 phút | 🟡 |
| 8 | Ablation Bar Chart | ❌ Cần tạo | 30 phút | 🟡 |
| 9 | Efficiency Scatter | ❌ Cần tạo | 30 phút | 🟡 |
| 10 | Augmentation Examples | ❌ Cần tạo | 30 phút | 🟢 |
| 11 | Cross-domain Heatmap | ❌ Cần tạo | 30 phút | 🟢 |
| 12 | SCIB Detail Diagram | ❌ Cần tạo | 2h | 🟢 |

**Tổng thời gian ước tính**: 
- 🔴 Ưu tiên cao: ~2.5h
- 🟡 Trung bình: ~2.5h  
- 🟢 Tùy chọn: ~3h
- **Tất cả: ~8h**

## Khuyến nghị tối thiểu cho journal submission

Thêm ít nhất **#1 + #2 + #5** (có sẵn hoặc nhanh) → 1.5h.
Lý tưởng thêm cả **#3 + #4 + #6** → +2.5h.

**Tổng hình sau khi bổ sung**: 10 hiện tại + 6 mới = **16 hình** (phù hợp journal 15-20 trang).
