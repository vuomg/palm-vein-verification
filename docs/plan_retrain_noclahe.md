# Plan: Chuẩn bị Dataset Internal không CLAHE (raw + tọa độ) + Train lại SCA-MobileNet

## Bối cảnh

Dataset `dataset_internal_openset` hiện tại đã qua full pipeline (GrabCut → Distance Transform → CLAHE → crop 128×128 PNG). Cần tạo phiên bản **raw + tọa độ ROI** (không qua bất kỳ xử lý nào) để train lại SCA-MobileNet, crop ROI on-the-fly khi train.

### Nguồn dữ liệu

- **Source**: `C:\project\auto\` — 1.549 users (autoUser1..autoUser1549 + 2 thư mục "Tung tay")
- **Mỗi user có** (index 1–10):
  - `img_*.raw` — ảnh gốc NIR, 307.200 bytes = 320×960 uint8 grayscale
  - `roi_*.txt` — tọa độ ROI: 8 số (x1,y1,x2,y2,x3,y3,x4,y4) = 4 điểm góc polygon
  - `quality_*.txt` — điểm chất lượng từ thiết bị (không dùng trong training)
  - `raw_feat_*.bin`, `ver_feat_*.bin`, `reg_template.bin` — features cũ (KHÔNG copy)

### Ví dụ tọa độ ROI

```
roi_1.txt: 294,334,117,348,129,526,307,512
→ 4 điểm: (294,334), (117,348), (129,526), (307,512)
→ Hệ tọa độ: ảnh sau resize 640×480 (height=640, width=480)
```

### Dataset hiện có (để so sánh)

| Dataset | Pipeline | Split |
|---------|----------|-------|
| `datasets/dataset_internal_openset/` | GrabCut → DT → CLAHE → 128×128 PNG | 1.084/465 open-set |

---

## Phase 1: Tạo dataset raw + tọa độ (không xử lý gì)

### 1.1 Script mới: `preprocessing/setup_internal_raw_dataset.py`

Tạo script mới (không sửa script cũ) với logic đơn giản:

```python
# Với mỗi user trong C:\project\auto:
#   1. Copy img_*.raw (giữ nguyên binary)
#   2. Copy roi_*.txt (tọa độ ROI)
# Split 70/30 open-set theo identity (seed=42)
```

### 1.2 Chạy

```bash
python preprocessing/setup_internal_raw_dataset.py \
    --input "C:/project/auto" \
    --output datasets/dataset_internal_raw_openset \
    --seed 42
```

### 1.3 Output: `datasets/dataset_internal_raw_openset/`

```
dataset_internal_raw_openset/
├── train/                          # 1.084 users
│   ├── autoUser1/
│   │   ├── img_1.raw               # 307.200 bytes, giữ nguyên
│   │   ├── img_2.raw
│   │   ├── ...
│   │   ├── img_10.raw
│   │   ├── roi_1.txt               # "294,334,117,348,129,526,307,512"
│   │   ├── roi_2.txt
│   │   ├── ...
│   │   └── roi_10.txt
│   ├── autoUser10/
│   └── ...
├── test/                           # 465 users
│   └── ...
└── split_info_openset.json         # metadata: seed, split, user lists
```

**Dung lượng ước tính**: 1.549 × 10 × (307.200 + ~50) ≈ **4.7 GB** (chủ yếu là .raw files)

### 1.4 Kiểm tra

- Verify 1.549 users chia đúng 1.084/465 (seed=42)
- Verify mỗi user có đủ 10 raw + 10 roi
- Verify raw file size = 307.200 bytes (không bị corrupt khi copy)
- So sánh split với `dataset_internal_openset` (phải giống nhau nếu cùng seed=42 + cùng danh sách user)

**Thời gian ước tính**: ~5–10 phút (chỉ copy file, không xử lý ảnh)

---

## Phase 2: Custom DataLoader cho raw + ROI

### 2.1 Tạo `RawROIDataset` class

File: `models/SCA_MobileNet/raw_dataset.py` (hoặc thêm vào file dataset hiện có)

```python
class RawROIDataset(Dataset):
    """Dataset đọc .raw files + crop ROI từ tọa độ roi_*.txt on-the-fly."""

    def __init__(self, data_dir, transform=None, target_size=(128, 128)):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.transform = transform
        self.samples = []  # List of (raw_path, roi_path, label)
        self.class_to_idx = {}

        # Scan user directories
        for idx, user_dir in enumerate(sorted(self.data_dir.iterdir())):
            if not user_dir.is_dir():
                continue
            self.class_to_idx[user_dir.name] = idx
            for raw_file in sorted(user_dir.glob("img_*.raw")):
                num = raw_file.stem.split('_')[1]  # "1", "2", ...
                roi_file = user_dir / f"roi_{num}.txt"
                if roi_file.exists():
                    self.samples.append((raw_file, roi_file, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_path, roi_path, label = self.samples[idx]

        # 1. Đọc raw → numpy
        raw_data = np.fromfile(str(raw_path), dtype=np.uint8)
        image = raw_data[:320 * 960].reshape(320, 960)
        left = image[:, 0:480]
        img = cv2.resize(left, (480, 640), interpolation=cv2.INTER_CUBIC)

        # 2. Parse ROI coordinates
        roi_text = roi_path.read_text().strip()
        coords = list(map(int, roi_text.split(',')))
        pts = np.array(coords).reshape(4, 2).astype(np.float32)

        # 3. Perspective transform → crop ROI
        dst_pts = np.array([
            [self.target_size[1]-1, 0],
            [0, 0],
            [0, self.target_size[0]-1],
            [self.target_size[1]-1, self.target_size[0]-1]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst_pts)
        roi = cv2.warpPerspective(img, M, self.target_size)

        # 4. Normalize to tensor
        if self.transform:
            roi = self.transform(roi)
        else:
            roi = torch.from_numpy(roi).float().unsqueeze(0) / 255.0

        return roi, label
```

### 2.2 Tích hợp vào `train.py`

Thêm flag `--raw-dataset` hoặc tự detect (kiểm tra có `.raw` files không):

```python
if args.raw_dataset or any((data_path / "train").rglob("*.raw")):
    from models.SCA_MobileNet.raw_dataset import RawROIDataset
    train_dataset = RawROIDataset(data_path / "train", transform=train_transform)
    test_dataset = RawROIDataset(data_path / "test", transform=test_transform)
else:
    # Pipeline hiện tại: ImageFolder cho PNG
    train_dataset = datasets.ImageFolder(...)
```

### 2.3 Lưu ý

- Perspective transform giữ nguyên toàn bộ thông tin trong vùng ROI
- Không CLAHE, không GrabCut, không Distance Transform
- Augmentation vẫn áp dụng bình thường (rotation, noise, etc.)
- BalancedBatchSampler vẫn hoạt động (16 classes × 4 samples)

**Thời gian ước tính**: ~1-2 giờ (viết + test DataLoader)

---

## Phase 3: Train SCA-MobileNet

### 3.1 Lệnh train

```bash
python train.py \
    --model sca_mobilenet \
    --dataset datasets/dataset_internal_raw_openset \
    --raw-dataset \
    --batch-size 16 \
    --epochs 100 \
    --lr 0.001 \
    --feature-dim 1024 \
    --eval-frequency 5 \
    --database default \
    --loss-type adacos_only \
    --sca-backbone mobilenetv3
```

### 3.2 Cấu hình (giữ nguyên như bản CLAHE)

| Tham số | Giá trị |
|---------|---------|
| Model | SCA-MobileNet (STN + CA + SPP) |
| Backbone | MobileNetV3-Small (cắt layer 10-11) |
| Loss | AdaCos (s=auto, m=0.35) |
| Optimizer | AdamW (lr=0.001, β₁=0.9, β₂=0.999) |
| Scheduler | MultiStepLR (milestones: [30, 60, 85], γ=0.1) |
| Batch | BalancedBatchSampler (16 classes × 4 samples = 64) |
| Warmup | 5 epochs (backbone frozen) |
| AMP | fp16 |
| Gradient clipping | max_norm=10.0 |
| Seed | 42 |
| Epochs | 100 |

### 3.3 Output

- `results/results_internal_raw_sca_mobilenet/`
  - `checkpoints/best_model_eer.pth`
  - `training_metrics.json`
  - `charts/` — biểu đồ loss, EER, TAR

**Thời gian ước tính**: ~2-3 giờ (100 epochs × ~1.5 phút/epoch)

---

## Phase 4: So sánh kết quả

### 4.1 So sánh CLAHE (GrabCut ROI) vs Raw (tọa độ ROI)

| Chỉ số | CLAHE + GrabCut ROI | Raw + Tọa độ ROI |
|--------|---------------------|-------------------|
| EER (%) | 0.90 | ? |
| TAR@0.01% | 98.00% | ? |
| TAR@0.1% | 99.10% | ? |
| AUC | 0.9985 | ? |
| d-prime | 5.12 | ? |

### 4.2 Phân tích

- **Nếu raw tốt hơn/tương đương**: Pipeline GrabCut + CLAHE không cần thiết → simplify, dùng tọa độ ROI có sẵn
- **Nếu raw kém hơn**: GrabCut + CLAHE đóng vai trò quan trọng → giữ trong pipeline
- **Lưu ý**: Sự khác biệt không chỉ do CLAHE mà còn do phương pháp crop ROI khác nhau:
  - Bản cũ: GrabCut segmentation → Distance Transform → ROI tự tính
  - Bản mới: Tọa độ ROI có sẵn từ thiết bị → Perspective Transform

---

## Tóm tắt

| Phase | Công việc | Thời gian |
|-------|----------|-----------|
| 1 | Script copy raw + tọa độ + split | ~10 phút |
| 2 | Custom DataLoader (RawROIDataset) | ~1-2 giờ |
| 3 | Train SCA-MobileNet | ~2-3 giờ |
| 4 | So sánh kết quả | ~10 phút |
| **Tổng** | | **~4-5 giờ** |

### Lưu ý quan trọng

- **Giữ nguyên .raw**: Không convert sang PNG/JPG → đảm bảo feature gốc 100%
- **Cùng seed=42** và **cùng split 70/30** → so sánh công bằng
- Tọa độ ROI từ `roi_*.txt` là output của thiết bị thu nhận (không phải GrabCut)
- Hệ tọa độ ROI: trên ảnh đã resize 640×480 (left half of 320×960 raw)
- Dung lượng dataset: ~4.7 GB (vs ~300 MB cho bản PNG 128×128)
