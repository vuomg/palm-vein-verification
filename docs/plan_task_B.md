# Task B: Plan - 12 Models trên Bộ Nội bộ

## Mục tiêu
Thêm bảng đầy đủ 12 models trên bộ dữ liệu nội bộ (1.549 ID) vào paper,
tương tự các bảng TONGJI/SCUT/VERA hiện có.

## Hiện trạng

### Bảng SOTA nội bộ hiện tại (chỉ 4 models)

| Model | Params (M) | EER (%) | TAR@0.01% | AUC |
|-------|-----------|---------|-----------|-----|
| FGFNet | 5.56 | 1.25 | 92.50% | 0.9970 |
| GSCL | 11.69 | 1.45 | 94.30% | 0.9950 |
| RSNet | 6.23 | 1.10 | 96.00% | 0.9980 |
| **SCA-MobileNet** | **3.19** | **0.90** | **98.00%** | **0.9985** |

### 8 Models thiếu trên bộ nội bộ

| # | Model | Params (M) | FLOPs (G) | Checkpoint tồn tại | training_metrics.json | Cần chạy lại |
|---|-------|-----------|-----------|--------------------|-----------------------|-------------|
| 1 | MPSNet | 2.99 | 0.15 | results_mpsnet/best_mspnet_model_eer.pth | NO | YES |
| 2 | Modified-DenseNet161 | 28.74 | 7.85 | results_eusipco2020/ | YES (eusipco2020_eusipco2020) | VERIFY |
| 3 | MobileNetV3 (Base) | 0.55 | 0.10 | results_mobilenetv3_base/ | NO | YES |
| 4 | EfficientNet-B0 | 5.29 | 0.40 | results_efficientnet_b0/ | NO | YES |
| 5 | ResNet50 | 25.23 | 4.12 | results_resnet50_gscl/ | NO | YES |
| 6 | Swin-Tiny | 28.29 | 4.49 | results_Swin-Tiny/ | YES (nghi ngờ) | VERIFY |
| 7 | DeiT-Tiny | 5.72 | 1.26 | results_DeiT-Tiny/ | YES (nghi ngờ) | VERIFY |
| 8 | MobileViT-S | 5.58 | 1.84 | results_MobileViT-S/ | YES (nghi ngờ) | VERIFY |

## Checklist

### Phase 1: Verify existing results
- [ ] Kiểm tra `results_eusipco2020_eusipco2020/training_metrics.json` — lấy best EER
- [ ] Kiểm tra `results_DeiT-Tiny/training_metrics.json` — lấy best EER
- [ ] Kiểm tra `results_MobileViT-S/training_metrics.json` — lấy best EER
- [ ] Kiểm tra `results_Swin-Tiny/training_metrics.json` — lấy best EER
- [ ] So sánh với kết quả trên TONGJI/SCUT — nếu hợp lý thì dùng, nếu bất thường thì retrain

### Phase 2: Retrain missing models (~6-8 giờ trên RTX 4050)
- [ ] Train MPSNet (`--model mpsnet`)
- [ ] Train MobileNetV3 Base (`--model sca_mobilenet --sca-backbone mobilenetv3 --no-stn --no-ca --no-spp`)
- [ ] Train EfficientNet-B0 (`--model sca_mobilenet --sca-backbone efficientnet_b0`)
- [ ] Train ResNet50 (`--model gscl --gscl-backbone resnet50`)
- [ ] Train các model khác nếu results hiện có không hợp lệ

### Phase 3: Thêm vào paper
- [ ] Tạo bảng 12 models trên bộ nội bộ (format giống tab:tongji_results)
- [ ] Thêm section phân tích kết quả
- [ ] Cập nhật SOTA table (tab:sota) thêm 8 models hoặc reference bảng mới
- [ ] Cập nhật abstract nếu cần

### Phase 4: Inference Benchmark
- [ ] Chạy `evaluation/inference_benchmark.py`
- [ ] Tạo bảng benchmark cho paper:

| Model | Params (M) | FLOPs (G) | Latency CPU (ms) | Latency GPU (ms) | FPS GPU | EER (%) |
|-------|-----------|-----------|-------------------|-------------------|---------|---------|
| SCA-MobileNet | 3.19 | 0.26 | ? | ? | ? | 0.90 |
| MPSNet | 2.99 | 0.15 | ? | ? | ? | ? |
| MobileNetV3 Base | 0.55 | 0.10 | ? | ? | ? | ? |
| ... | ... | ... | ... | ... | ... | ... |

### Phase 5: Bubble Chart
- [ ] Chạy `tools/generate_bubble_chart.py`
- [ ] Output: `paper/fig_efficiency_bubble.png`
- [ ] Thêm figure vào paper

## Thông số 12 Models (từ paper Table 3)

| Model | Params (M) | FLOPs (G) | Input Size | Emb Dim |
|-------|-----------|-----------|------------|---------|
| SCA-MobileNet | 3.19 | 0.26 | 224×224 | 1024 |
| MobileNetV3 Base | 0.55 | 0.10 | 224×224 | 1024 |
| MPSNet | 2.99 | 0.15 | 224×224 | 1024 |
| EfficientNet-B0 | 5.29 | 0.40 | 224×224 | 1024 |
| DeiT-Tiny | 5.72 | 1.26 | 224×224 | 1024 |
| FGFNet | 5.56 | 6.37 | 256×256 | 640 |
| MobileViT-S | 5.58 | 1.84 | 224×224 | 1024 |
| RSNet | 6.23 | 1.17 | 224×224 | 1024 |
| GSCL | 11.69 | 1.82 | 224×224 | 512 |
| ResNet50 | 25.23 | 4.12 | 224×224 | 1024 |
| Modified-DenseNet161 | 28.74 | 7.85 | 224×224 | 1024 |
| Swin-Tiny | 28.29 | 4.49 | 224×224 | 1024 |

## Ước lượng thời gian

| Phase | Thời gian |
|-------|-----------|
| Phase 1: Verify | ~15 phút |
| Phase 2: Retrain 4-8 models | ~6-8 giờ |
| Phase 3: Paper edit | ~30 phút |
| Phase 4: Benchmark | ~30 phút |
| Phase 5: Bubble chart | ~10 phút |
| **Tổng** | **~8-10 giờ** |
