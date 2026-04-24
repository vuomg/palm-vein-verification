# TODO: Việc chưa làm cho Paper

## 1. Train TONGJI thiếu 6 model ⏳

Paper SCUT/VERA có 12 model, TONGJI chỉ có 6. Cần train thêm:

```bash
# 1. ResNet50 (GSCL framework, cần conda env gscl)
cd models/GSCL-PyTorch/vein_feature_learning
python train_palmvein_fusionaug.py \
    --trainset datasets/TONGJI_dataset_openset/train \
    --testset datasets/TONGJI_dataset_openset/test \
    --dataset_name tongji_resnet50 \
    --network resnet50 --loss fusionloss \
    --max_epoch 100 --p 16 --k 4 --lr 0.01 --eval_freq 5 --seed 42

# 2. MobileNetV3-Base
python train.py --model sca_mobilenet --no-stn --no-ca --no-spp \
    --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_mobilenetv3_base \
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 \
    --batch-size 16 --epochs 100 --lr 0.001 \
    --database TJ_PV --eval-frequency 5 --seed 42

# 3. EfficientNet-B0
python train.py --model sca_mobilenet --sca-backbone efficientnet_b0 \
    --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_efficientnet_b0 \
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 \
    --batch-size 16 --epochs 100 --lr 0.001 \
    --database TJ_PV --eval-frequency 5 --seed 42

# 4. Swin-Tiny
python train.py --model sca_mobilenet --sca-backbone swin_tiny \
    --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_swin_tiny \
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 \
    --batch-size 16 --epochs 100 --lr 0.001 \
    --database TJ_PV --eval-frequency 5 --seed 42

# 5. DeiT-Tiny
python train.py --model sca_mobilenet --sca-backbone deit_tiny \
    --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_deit_tiny \
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 \
    --batch-size 16 --epochs 100 --lr 0.001 \
    --database TJ_PV --eval-frequency 5 --seed 42

# 6. MobileViT-S
python train.py --model sca_mobilenet --sca-backbone mobilevit_s \
    --dataset datasets/TONGJI_dataset_openset --output-dir results_tongji_mobilevit_s \
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 \
    --batch-size 16 --epochs 100 --lr 0.001 \
    --database TJ_PV --eval-frequency 5 --seed 42
```

**Thời gian**: ~12-15h GPU

---

## 2. Chạy Cross-Domain VERA ↔ SCUT ⏳

Script đã tạo, chưa chạy. 12 model × 2 chiều = 24 evaluations.

```bash
# Chạy tất cả
scripts\run_cross_domain_vera_scut.bat

# Hoặc từng chiều
python evaluation/cross_domain_vera_scut.py --direction vera_to_scut --all
python evaluation/cross_domain_vera_scut.py --direction scut_to_vera --all
```

**Kết quả**: `results/results_cross_domain_vera_scut/`
**Thời gian**: ~5-6h GPU

---

## 3. Sửa EER inconsistency trong LaTeX ✅ DONE

File: `paper/journal/palm_vein_journal.tex`

| Bảng | Dòng | Hiện tại | Vấn đề |
|------|------|----------|--------|
| Ablation Config 8 | L350 | EER = **0.90**, TAR = **98.00%** | |
| Comparison Table | L387 | EER = **0.89**, TAR = **97.99%** | Khác ablation |
| Summary Table | L624 | EER = **0.90**, TAR = **98.00%** | Khớp ablation |

**Cần**: Chọn 1 giá trị thống nhất cho cả 3 bảng (hoặc chạy lại 1 lần rồi dùng kết quả đó).

---

## 4. TONGJI 6 model cũ thiếu `training_metrics.json` ⏳

7 kết quả TONGJI cũ chỉ có `.pth`, không có metrics JSON (pipeline cũ):

```
results_tongji_mpsnet_mpsnet/
results_tongji_densenet161_eusipco2020/
results_tongji_gscl_gscl/
results_tongji_rsnet_rsnet/
results_tongji_fgfnet_fgfnet/
results_tongji_sca_mobilenet/
results_tongji_sca_sca_mobilenet_sca_mobilenet/
```

**Giải pháp**: Train lại 6 model này trên TONGJI (không dùng `--no-checkpoint`) để có cả `.pth` + `training_metrics.json` thống nhất. Hoặc chỉ cần số liệu từ paper thì bỏ qua.

---

## 5. (Tùy chọn) Internal thiếu 3 transformer ⏳

Paper Internal chỉ có 9 model, SCUT/VERA có 12. Nếu muốn thống nhất:

```bash
# Swin-Tiny
python train.py --model sca_mobilenet --sca-backbone swin_tiny \
    --dataset datasets/final_dataset_openset --output-dir results_internal_swin_tiny \
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 \
    --batch-size 16 --epochs 100 --lr 0.001 --eval-frequency 5 --seed 42

# DeiT-Tiny
python train.py --model sca_mobilenet --sca-backbone deit_tiny \
    --dataset datasets/final_dataset_openset --output-dir results_internal_deit_tiny \
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 \
    --batch-size 16 --epochs 100 --lr 0.001 --eval-frequency 5 --seed 42

# MobileViT-S
python train.py --model sca_mobilenet --sca-backbone mobilevit_s \
    --dataset datasets/final_dataset_openset --output-dir results_internal_mobilevit_s \
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 \
    --batch-size 16 --epochs 100 --lr 0.001 --eval-frequency 5 --seed 42
```

**Thời gian**: ~6-8h GPU

---

## Tổng thời gian ước tính

| Việc | GPU time | Ưu tiên |
|------|----------|---------|
| Train TONGJI 6 model mới | ~12-15h | 🔴 Cao |
| Cross-domain VERA↔SCUT | ~5-6h | 🔴 Cao (thầy yêu cầu) |
| Sửa EER LaTeX | 10 phút | ✅ Done |
| Train lại TONGJI 6 model cũ (lấy JSON) | ~12h | 🟡 Trung bình |
| Train Internal 3 transformer | ~6-8h | 🟢 Tùy chọn |
| **Tổng (bắt buộc)** | **~17-21h** | |
