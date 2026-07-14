"""
Grad-CAM Visualization — SCA-MobileNet vs ViT Baselines
So sánh attention map: model nào focus đúng vùng tĩnh mạch hơn
Dùng cho paper Q1 — Qualitative Analysis
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import cv2
import os
from pathlib import Path

# pip install grad-cam
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
try:
    # Optional: improves Grad-CAM for transformer models
    from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform, swin_reshape_transform
except Exception:
    vit_reshape_transform = None
    swin_reshape_transform = None

# ============================================================
# CẤU HÌNH
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'models'))
# Prefer local project dataset first; fallback to legacy absolute path.
_dataset_candidates = [
    PROJECT_ROOT / "datasets" / "final_dataset_openset",
    PROJECT_ROOT / "final_dataset_openset",
]
DATASET_DIR = next((p for p in _dataset_candidates if p.exists()), _dataset_candidates[0])
RESULTS_ROOT = PROJECT_ROOT / "results" / "results_sca_v2_sca_mobilenet"

OUTPUT_DIR = str(RESULTS_ROOT / "gradcam")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _first_existing_image(dir_candidates, patterns):
    """Return first matching image path across candidate identity folders."""
    for d in dir_candidates:
        if not d.exists() or not d.is_dir():
            continue
        for pat in patterns:
            matches = sorted(d.glob(pat))
            if matches:
                return str(matches[0])
    return None


def resolve_sample_images(split="test", identity_indices=(0, 20, 34)):
    # This repo's README describes the open-set folder structure as:
    #   {train,test}/identity_XXX/{images...}
    # Some older scripts used id{N}; we support both.
    patterns = [
        # common ROI naming
        "roi_1.png",
        "roi_*.png",
        "roi_1.jpg",
        "roi_*.jpg",
        # common image naming
        "img1.png",
        "img*.png",
        # any image fallback in that identity folder
        "*.png",
        "*.jpg",
        "*.bmp",
        # and one-level deep if identity folder has subfolders
        "**/*.png",
        "**/*.jpg",
        "**/*.bmp",
    ]

    images = []
    for idx in identity_indices:
        candidates = [
            DATASET_DIR / split / f"identity_{idx + 1:03d}",
            DATASET_DIR / split / f"id{idx}",
            DATASET_DIR / split / f"identity_{idx:03d}",
        ]
        img = _first_existing_image(candidates, patterns)
        if img is not None:
            images.append(img)

    # Final fallback: pick the first 3 images anywhere in the split folder.
    if len(images) < len(identity_indices):
        allowed = {".png", ".jpg", ".jpeg", ".bmp"}
        all_imgs = [p for p in (DATASET_DIR / split).glob("**/*") if p.is_file() and p.suffix.lower() in allowed]
        all_imgs = sorted(all_imgs)
        for p in all_imgs:
            if len(images) >= len(identity_indices):
                break
            if str(p) not in images:
                images.append(str(p))

    return images


SAMPLE_IMAGES = resolve_sample_images()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBEDDING_DIM = 1024
NUM_CLASSES = 1084

GRADCAM_MODEL_CONFIGS = {
    "RSNet": {
        "checkpoint": "results/results_rsnet/best_rsnet_model_eer.pth",
        "input_size": 224,
        "input_channels": 3,
    },
    "SCA-MobileNet (Ours)": {
        "checkpoint": "results/results_sca_v2_sca_mobilenet/best_sca_mobilenet_model_eer.pth",
        "input_size": 224,
        "input_channels": 3,
    },
    "MPSNet": {
        "checkpoint": "results/results_hutech_cnpv1549/mpsnet_mpsnet/best_mpsnet_model_eer.pth",
        "input_size": 224,
        "input_channels": 1,
    },
    "FGFNet": {
        "checkpoint": "results/results_fgfnet/best_fgfnet_model_eer.pth",
        "input_size": 256,
        "input_channels": 3,
    },
    "GSCL (ResNet-18)": {
        "checkpoint": "models/GSCL-PyTorch/vein_feature_learning/results/palmvein_resnet18/checkpoints/best_model_seed42.pth",
        "input_size": 224,
        "input_channels": 3,
    },
    "DeiT-Tiny": {
        "checkpoint": "results/results_DeiT-Tiny/best_DeiT-Tiny_model_eer.pth",
        "input_size": 224,
        "input_channels": 3,
    },
    "MobileViT-S": {
        "checkpoint": "results/results_MobileViT-S/best_MobileViT-S_model_eer.pth",
        "input_size": 256,
        "input_channels": 3,
    },
    "Swin-Tiny": {
        "checkpoint": "results/results_Swin-Tiny/best_Swin-Tiny_model_eer.pth",
        "input_size": 224,
        "input_channels": 3,
    },
    "EfficientNet-B0": {
        "checkpoint": "results/results_efficientnet_b0/best_efficientnet_b0_model_eer.pth",
        "input_size": 224,
        "input_channels": 3,
    },
    "MobileNetV3-Small": {
        "checkpoint": "results/results_mobilenetv3_base/best_mobilenetv3_base_model_eer.pth",
        "input_size": 224,
        "input_channels": 3,
    },
}


# ============================================================
# HELPER: Load và preprocess ảnh
# ============================================================
def load_image(path, size=224, channels=3):
    img = Image.open(path).convert("RGB").resize((size, size))
    img_np = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy((img_np - mean) / std).permute(2, 0, 1).unsqueeze(0).float()
    if channels == 1:
        tensor = tensor[:, :1, :, :] * 0.299 + tensor[:, 1:2, :, :] * 0.587 + tensor[:, 2:3, :, :] * 0.114
    return img_np, tensor


# ============================================================
# HELPER: Lấy target layer cho từng model
# ============================================================
def find_last_conv_layer(model: nn.Module):
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


def get_target_layer(model, model_name):
    """Return the appropriate layer for Grad-CAM hook based on model architecture."""
    name = model_name.lower()

    # --- RSNet: rleb (Residual Local Enhancement Block) before global pooling ---
    if "rsnet" in name:
        if hasattr(model, "rleb"):
            return [model.rleb]
        if hasattr(model, "stage4_down"):
            return [model.stage4_down]

    # --- SCA-MobileNet: last layer in self.features ---
    if "sca" in name and hasattr(model, "features"):
        return [model.features[-1]]

    # --- MPSNet: last conv ---
    if "mpsnet" in name or "msnet" in name:
        last_conv = find_last_conv_layer(model)
        if last_conv is not None:
            return [last_conv]

    # --- FGFNet: last conv ---
    if "fgfnet" in name or "fgf" in name:
        last_conv = find_last_conv_layer(model)
        if last_conv is not None:
            return [last_conv]

    # --- GSCL (ResNet-18): layer4 (index 7 in backbone Sequential) ---
    if "gscl" in name or "resnet-18" in name:
        backbone = getattr(model, "backbone", model)
        if hasattr(backbone, "layer4"):
            return [backbone.layer4[-1]]
        try:
            return [backbone[7]]
        except (IndexError, TypeError):
            pass
        last_conv = find_last_conv_layer(model)
        if last_conv is not None:
            return [last_conv]

    # --- EUSIPCO DenseNet161: features.denseblock4 ---
    if "eusipco" in name or "densenet" in name:
        if hasattr(model, "features") and hasattr(model.features, "denseblock4"):
            return [model.features.denseblock4]
        if hasattr(model, "features"):
            return [model.features[-1]]
        last_conv = find_last_conv_layer(model)
        if last_conv is not None:
            return [last_conv]

    # --- DeiT / ViT (including SCATransformerBackbone wrapping timm) ---
    if "deit" in name or ("vit" in name and "mobilevit" not in name):
        if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
            return [model.backbone.blocks[-1].norm1]
        if hasattr(model, "blocks"):
            return [model.blocks[-1].norm1]

    # --- MobileViT (timm or SCATransformerBackbone) ---
    if "mobilevit" in name:
        backbone = getattr(model, "backbone", model)
        if hasattr(backbone, "final_conv"):
            return [backbone.final_conv]
        if hasattr(backbone, "stages"):
            try:
                last_stage = backbone.stages[-1]
                if hasattr(last_stage, "__getitem__") and len(last_stage) > 0:
                    last_block = last_stage[-1]
                    if hasattr(last_block, "conv_fusion"):
                        return [last_block.conv_fusion]
                return [last_stage]
            except Exception:
                pass
        last_conv = find_last_conv_layer(backbone)
        if last_conv is not None:
            return [last_conv]

    # --- Swin (timm or SCATransformerBackbone) ---
    if "swin" in name:
        backbone = getattr(model, "backbone", model)
        if hasattr(backbone, "layers"):
            try:
                return [backbone.layers[-1].blocks[-1].norm1]
            except Exception:
                pass

    # --- EfficientNet-B0: features[-1] ---
    if "efficientnet" in name:
        if hasattr(model, "features"):
            return [model.features[-1]]

    # --- MobileNetV3-Small (non-SCA): features[-1] ---
    if "mobilenetv3" in name or "mobilenet" in name:
        if hasattr(model, "features"):
            return [model.features[-1]]

    # --- ResNet-50: layer4 ---
    if "resnet" in name and "50" in name:
        if hasattr(model, "layer4"):
            return [model.layer4[-1]]

    # Fallback: last conv layer
    last_conv = find_last_conv_layer(model)
    if last_conv is not None:
        return [last_conv]
    return [list(model.children())[-1]]


def _reshape_tokens_to_spatial(tensor):
    """
    Convert transformer token output to NCHW for CAM.
    Supports [B, N, C] and [B, C, H, W] tensors.
    """
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim != 3:
        raise ValueError(f"Unsupported tensor shape for reshape: {tuple(tensor.shape)}")

    b, n, c = tensor.shape
    # Remove CLS token if present (common for ViT/DeiT).
    side = int(np.sqrt(n))
    if side * side != n and n > 1:
        n = n - 1
        tensor = tensor[:, 1:, :]
        side = int(np.sqrt(n))
    if side * side != n:
        raise ValueError(f"Cannot infer spatial size from token count={n}")

    tensor = tensor.reshape(b, side, side, c)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor


def _get_reshape_transform(model_name):
    name = model_name.lower()
    if "mobilevit" in name:
        return None
    if "deit" in name or ("vit" in name and "mobilevit" not in name):
        return _reshape_tokens_to_spatial
    if "swin" in name:
        return swin_reshape_transform if swin_reshape_transform else _reshape_tokens_to_spatial
    return None


# ============================================================
# MAIN: Tạo Grad-CAM comparison figure
# ============================================================
DISPLAY_SIZE = 512
CAM_COLORMAP = cv2.COLORMAP_JET
CAM_ALPHA = 0.45


def _overlay_cam(img_np, grayscale_cam):
    """Apply a consistent JET colormap overlay on the image.

    ``grayscale_cam`` is expected in [0, 1].  The same colormap and alpha
    are used across every model so that colors are directly comparable.
    """
    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(grayscale_cam, (w, h))
    cam_resized = np.clip(cam_resized, 0, 1)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), CAM_COLORMAP)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blended = (1 - CAM_ALPHA) * img_np + CAM_ALPHA * heatmap
    return np.clip(blended, 0, 1), cam_resized


def generate_gradcam_figure(models_dict, image_paths, labels=None):
    """
    Generate Grad-CAM++ comparison figure across all models.

    Args:
        models_dict: {name: {"model": nn.Module, "input_size": int, "input_channels": int}}
        image_paths: list of image file paths
        labels: optional row labels
    """
    n_images = len(image_paths)
    n_models = len(models_dict)
    if n_images == 0:
        raise ValueError(
            f"No sample images found under dataset: {DATASET_DIR}. "
            "Please verify dataset path and split structure."
        )
    img_labels = labels or [f"Sample {i+1}" for i in range(n_images)]

    col_width = 1.65
    row_height = 1.75
    n_cols = n_models + 1
    fig_w = col_width * n_cols + 0.6
    fig_h = row_height * n_images + 0.7
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = gridspec.GridSpec(
        n_images, n_cols + 1,
        width_ratios=[1] * n_cols + [0.04],
        hspace=0.30, wspace=0.06,
        left=0.01, right=0.95, top=0.88, bottom=0.02,
    )

    all_cam_images = []

    for row, (img_path, img_label) in enumerate(zip(image_paths, img_labels)):
        display_np, _ = load_image(img_path, size=DISPLAY_SIZE)

        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(display_np, interpolation="lanczos")
        ax.set_title("Original" if row == 0 else "", fontsize=11, fontweight="bold")
        ax.set_ylabel(img_label, fontsize=11, fontweight="bold")
        ax.axis("off")

        for col, (model_name, model_info) in enumerate(models_dict.items()):
            model = model_info["model"].to(DEVICE).eval()
            input_size = model_info.get("input_size", 224)
            input_channels = model_info.get("input_channels", 3)

            _, img_tensor = load_image(img_path, size=input_size, channels=input_channels)
            img_tensor = img_tensor.to(DEVICE)

            target_layers = get_target_layer(model, model_name)
            reshape_transform = _get_reshape_transform(model_name)

            grayscale_cam = None
            for cam_cls in (GradCAMPlusPlus, GradCAM):
                try:
                    cam = cam_cls(model=model, target_layers=target_layers,
                                  reshape_transform=reshape_transform)
                    grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0]
                    break
                except Exception as e:
                    if cam_cls is GradCAMPlusPlus:
                        print(f"  GradCAM++ failed for {model_name}: {e}. Retrying GradCAM...")

            ax = fig.add_subplot(gs[row, col + 1])
            if grayscale_cam is not None:
                visualization, cam_for_cbar = _overlay_cam(display_np, grayscale_cam)
                im = ax.imshow(visualization, interpolation="lanczos")
                all_cam_images.append(cam_for_cbar)
            else:
                print(f"  GradCAM failed for {model_name}")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=12, color="#999", transform=ax.transAxes)
            if row == 0:
                ax.set_title(model_name, fontsize=11, fontweight="bold")
            ax.axis("off")

    cbar_ax = fig.add_subplot(gs[:, -1])
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Attention intensity", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle("Grad-CAM++ Attention Visualization — All Models",
                 fontsize=14, fontweight="bold", y=0.97)

    for fmt in ("png", "pdf"):
        out = os.path.join(OUTPUT_DIR, f"gradcam_comparison.{fmt}")
        fig.savefig(out, dpi=600 if fmt == "png" else 300,
                    bbox_inches="tight", pad_inches=0.05)
        print(f"Saved: {out}")
    plt.close()
    return os.path.join(OUTPUT_DIR, "gradcam_comparison.png")


# ============================================================
# BONUS: STN Deformation Visualization (đã có trong paper)
# Tạo figure đẹp hơn với alignment overlay
# ============================================================
def visualize_stn_alignment(sca_model, image_paths, output_dir):
    """
    Visualize STN input vs output để chứng minh geometric alignment
    """
    sca_model = sca_model.to(DEVICE).eval()
    fig, axes = plt.subplots(len(image_paths), 2,
                             figsize=(6, 3 * len(image_paths)))

    for i, path in enumerate(image_paths):
        img_np, img_tensor = load_image(path)
        img_tensor = img_tensor.to(DEVICE)

        with torch.no_grad():
            # Hook vào STN output
            # TODO: điều chỉnh tùy theo cách bạn implement STN trong SCAMobileNet
            aligned = sca_model.stn(img_tensor)

        # Original
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Before STN" if i == 0 else "")
        axes[i, 0].axis("off")

        # Aligned
        aligned_np = aligned.squeeze().permute(1, 2, 0).cpu().numpy()
        aligned_np = (aligned_np - aligned_np.min()) / (aligned_np.max() - aligned_np.min())
        axes[i, 1].imshow(aligned_np)
        axes[i, 1].set_title("After STN" if i == 0 else "")
        axes[i, 1].axis("off")

    plt.suptitle("Spatial Transformer Network Alignment", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "stn_alignment.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# MODEL WRAPPERS (single-tensor output for GradCAM)
# ============================================================
class _EvalWrapper(nn.Module):
    """Wraps models that return tuples to return only the first element."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, tuple) else out


class _BackboneOnlyWrapper(nn.Module):
    """Uses only the backbone for GradCAM (skips head that may fail on 4D tensors)."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self, x):
        h = self.backbone(x)
        return h.flatten(1)


class _EusipcoWrapper(nn.Module):
    """EUSIPCO DenseNet161 uses forward(x, train=False) for inference."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x, train=False)


class _FGFNetWrapper(nn.Module):
    """FGFNet get_embedding() for feature-level GradCAM."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model.get_embedding(x)


# ============================================================
# BUILD ALL MODELS FROM CHECKPOINTS
# ============================================================
def _load_state_dict(checkpoint_path):
    """Load checkpoint and extract state_dict regardless of format."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "model" in ckpt:
            return ckpt["model"]
    return ckpt


def build_all_gradcam_models():
    """Build all 12 models from trained checkpoints for GradCAM."""
    import sys
    import importlib.util

    models = {}

    for name, cfg in GRADCAM_MODEL_CONFIGS.items():
        ckpt_path = cfg["checkpoint"]
        if ckpt_path is None:
            print(f"  [SKIP] {name}: no checkpoint configured")
            continue
        ckpt_full = str(PROJECT_ROOT / ckpt_path)
        if not os.path.isfile(ckpt_full):
            print(f"  [SKIP] {name}: checkpoint not found ({ckpt_full})")
            continue

        try:
            state_dict = _load_state_dict(ckpt_full)
            model = None

            # --- RSNet ---
            if name == "RSNet":
                from RSNet.model import RSNet
                model = RSNet(feature_dim=EMBEDDING_DIM, in_channels=3, dropout_rate=0.3)
                model.load_state_dict(state_dict, strict=False)

            # --- SCA-MobileNet (Ours) ---
            elif name == "SCA-MobileNet (Ours)":
                from SCA_MobileNet.model import SCAMobileNet
                model = SCAMobileNet(
                    embedding_size=EMBEDDING_DIM, class_size=NUM_CLASSES,
                    pretrained=False, only_embeddings=True,
                    use_stn=True, use_ca=True, use_spp=True, dropout=0.3,
                    use_bottleneck=False, ca_reduction=32,
                )
                model.load_state_dict(state_dict, strict=False)
                model = _EvalWrapper(model)

            # --- MPSNet ---
            elif name == "MPSNet":
                from MPSNet_2022.model_pytorch import MPSNet
                model = MPSNet(feature_dim=EMBEDDING_DIM, input_channels=1, dropout=0.3)
                model.load_state_dict(state_dict, strict=False)

            # --- FGFNet ---
            elif name == "FGFNet":
                from FGFNet.model import MobileViT_FFC_ATTN_FFTSA
                model = MobileViT_FFC_ATTN_FFTSA(image_size=(256, 256), num_classes=NUM_CLASSES)
                model.load_state_dict(state_dict, strict=False)
                model = _FGFNetWrapper(model)

            # --- GSCL (ResNet-18) ---
            elif name == "GSCL (ResNet-18)":
                gscl_path = str(PROJECT_ROOT / "models" / "GSCL-PyTorch" / "vein_feature_learning")
                if gscl_path not in sys.path:
                    sys.path.insert(0, gscl_path)
                from models.models import ResNets
                full_model = ResNets(backbone="resnet18", head_type="cls_norm", num_classes=NUM_CLASSES)
                full_model.load_state_dict(state_dict, strict=False)
                model = _BackboneOnlyWrapper(full_model.backbone)

            # --- EUSIPCO-DenseNet161 ---
            elif name == "EUSIPCO-DenseNet161":
                eusipco_dir = PROJECT_ROOT / "models" / "Modified_Densenet161_2021"
                spec = importlib.util.spec_from_file_location(
                    "modified_models", eusipco_dir / "models" / "modified_models.py"
                )
                modified_models = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(modified_models)
                model = modified_models.DenseNet161_Modified(
                    embedding_size=EMBEDDING_DIM, class_size=NUM_CLASSES,
                    pretrained=False, only_embeddings=True, l2_normed=True
                )
                model.load_state_dict(state_dict, strict=False)
                model = _EusipcoWrapper(model)

            # --- DeiT-Tiny / MobileViT-S / Swin-Tiny (SCATransformerBackbone) ---
            elif name in ("DeiT-Tiny", "MobileViT-S", "Swin-Tiny"):
                import timm
                backbone_map = {
                    "DeiT-Tiny": ("deit_tiny_patch16_224.fb_in1k", "deit_tiny"),
                    "MobileViT-S": ("mobilevit_s.cvnets_in1k", "mobilevit_s"),
                    "Swin-Tiny": ("swin_tiny_patch4_window7_224.ms_in1k", "swin_tiny"),
                }
                timm_name, _ = backbone_map[name]
                backbone = timm.create_model(timm_name, pretrained=False, num_classes=0, global_pool="avg")
                backbone_sd = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
                backbone.load_state_dict(backbone_sd, strict=False)
                model = backbone

            # --- EfficientNet-B0 ---
            elif name == "EfficientNet-B0":
                from torchvision.models import efficientnet_b0
                model = efficientnet_b0(weights=None)
                model.classifier = nn.Identity()
                model.load_state_dict(state_dict, strict=False)

            # --- MobileNetV3-Small ---
            elif name == "MobileNetV3-Small":
                from torchvision.models import mobilenet_v3_small
                model = mobilenet_v3_small(weights=None)
                model.classifier = nn.Identity()
                feat_sd = {k: v for k, v in state_dict.items() if k.startswith("features.")}
                model.load_state_dict(feat_sd, strict=False)

            if model is not None:
                model.eval()
                models[name] = {
                    "model": model,
                    "input_size": cfg["input_size"],
                    "input_channels": cfg["input_channels"],
                }
                n_params = sum(p.numel() for p in model.parameters())
                print(f"  [OK]   {name} ({n_params/1e6:.2f}M params)")

        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    return models


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Grad-CAM++ Visualization — All 12 Models (trained checkpoints)")
    print("=" * 70)

    print("\nLoading models...")
    models_dict = build_all_gradcam_models()
    print(f"\nLoaded {len(models_dict)}/{len(GRADCAM_MODEL_CONFIGS)} models")

    image_paths = SAMPLE_IMAGES
    labels = [
        "Normal (ID-0)",
        "Pose Failure (ID-20)",
        "Illumination Failure (ID-34)",
    ]

    if models_dict and image_paths:
        print("\nGenerating Grad-CAM++ comparison figure...")
        generate_gradcam_figure(models_dict, image_paths, labels)
        print("Done!")
    else:
        if not models_dict:
            print("\nNo models loaded. Check checkpoint paths.")
        if not image_paths:
            print(f"\nNo sample images found under: {DATASET_DIR}")
