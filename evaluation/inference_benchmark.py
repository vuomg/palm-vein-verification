"""
Inference Benchmark — SCA-MobileNet vs All Baselines
Measures latency (ms/sample), FPS, and parameter count on CPU and GPU.
Used for paper — Computational Complexity Analysis.
"""

import torch
import torch.nn as nn
import time
import sys
import numpy as np
import json
import os
import importlib.util
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'models'))
OUTPUT_DIR = str(PROJECT_ROOT / "benchmark_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_WARMUP = 50
N_RUNS = 500
DEVICE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_CPU = torch.device("cpu")

EMBEDDING_DIM = 1024
NUM_CLASSES = 100  # dummy value for models that require it


# ============================================================
# LATENCY MEASUREMENT
# ============================================================
def measure_latency(model, device, input_size, n_warmup=N_WARMUP, n_runs=N_RUNS):
    model = model.to(device).eval()
    dummy = torch.randn(input_size).to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "fps": float(1000 / np.mean(latencies)),
        "device": str(device),
    }


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_flops(model, input_size):
    """Estimate FLOPs using thop if available, otherwise return None."""
    try:
        from thop import profile
        dummy = torch.randn(input_size)
        flops, _ = profile(model.cpu(), inputs=(dummy,), verbose=False)
        return flops
    except Exception:
        return None


# ============================================================
# MODEL WRAPPERS (to ensure single-tensor output for benchmarking)
# ============================================================
class EvalWrapper(nn.Module):
    """Wraps models that return tuples in eval mode to return only embeddings."""
    def __init__(self, model, output_mode='first'):
        super().__init__()
        self.model = model
        self.output_mode = output_mode

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]
        return out


class EusipcoEvalWrapper(nn.Module):
    """EUSIPCO2020 models use forward(x, train=False) for inference."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, train=False)


class FGFNetEmbeddingWrapper(nn.Module):
    """FGFNet get_embedding() for fair embedding-level comparison."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.get_embedding(x)


# ============================================================
# BUILD ALL MODELS
# ============================================================
def build_models():
    """
    Returns dict of {name: (model, input_size)} for all models to benchmark.
    Models are in eval mode and wrapped to produce single-tensor output.
    """
    models = {}

    # ------------------------------------------------------------------
    # 1. RSNet (Paper model)
    # ------------------------------------------------------------------
    try:
        from RSNet.model import RSNet
        rsnet = RSNet(feature_dim=EMBEDDING_DIM, in_channels=3, pretrained=False)
        rsnet.eval()
        models["RSNet"] = (rsnet, (1, 3, 224, 224))
        print("  [OK] RSNet")
    except Exception as e:
        print(f"  [SKIP] RSNet: {e}")

    # ------------------------------------------------------------------
    # 2. SCA-MobileNet (Our model)
    # ------------------------------------------------------------------
    try:
        from SCA_MobileNet.model import SCAMobileNet
        sca = SCAMobileNet(
            embedding_size=EMBEDDING_DIM,
            class_size=NUM_CLASSES,
            pretrained=False,
            only_embeddings=True,
            l2_normed=True,
            use_stn=True,
            use_ca=True,
            use_spp=True,
            dropout=0.5
        )
        sca.eval()
        models["SCA-MobileNet (Ours)"] = (EvalWrapper(sca), (1, 3, 224, 224))
        print("  [OK] SCA-MobileNet (Ours)")
    except Exception as e:
        print(f"  [SKIP] SCA-MobileNet: {e}")

    # ------------------------------------------------------------------
    # 3. MPSNet
    # ------------------------------------------------------------------
    try:
        from MPSNet_2022.model_pytorch import MPSNet
        mpsnet = MPSNet(feature_dim=EMBEDDING_DIM, input_channels=1, dropout=0.2)
        mpsnet.eval()
        models["MPSNet"] = (mpsnet, (1, 1, 224, 224))
        print("  [OK] MPSNet")
    except Exception as e:
        print(f"  [SKIP] MPSNet: {e}")

    # ------------------------------------------------------------------
    # 4. FGFNet (MobileViT + FFC + FFT)
    # ------------------------------------------------------------------
    try:
        from FGFNet.model import MobileViT_FFC_ATTN_FFTSA
        fgfnet = MobileViT_FFC_ATTN_FFTSA(
            image_size=(256, 256),
            num_classes=NUM_CLASSES
        )
        fgfnet.eval()
        models["FGFNet"] = (FGFNetEmbeddingWrapper(fgfnet), (1, 3, 256, 256))
        print("  [OK] FGFNet")
    except Exception as e:
        print(f"  [SKIP] FGFNet: {e}")

    # ------------------------------------------------------------------
    # 5. GSCL (ResNet-18 backbone)
    # ------------------------------------------------------------------
    try:
        gscl_path = os.path.join(str(PROJECT_ROOT), 'models', 'GSCL-PyTorch', 'vein_feature_learning')
        if gscl_path not in sys.path:
            sys.path.insert(0, gscl_path)
        from models.models import ResNets
        gscl = ResNets(
            backbone='resnet18',
            head_type='cls_norm',
            num_classes=NUM_CLASSES
        )
        gscl.eval()
        models["GSCL (ResNet-18)"] = (EvalWrapper(gscl), (1, 3, 256, 256))
        print("  [OK] GSCL (ResNet-18)")
    except Exception as e:
        print(f"  [SKIP] GSCL: {e}")

    # ------------------------------------------------------------------
    # 6. EUSIPCO2020 — DenseNet-161
    # ------------------------------------------------------------------
    try:
        eusipco_dir = PROJECT_ROOT / 'models' / 'Modified_Densenet161_2021'
        spec = importlib.util.spec_from_file_location(
            "modified_models",
            eusipco_dir / 'models' / 'modified_models.py'
        )
        modified_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modified_models)

        densenet = modified_models.DenseNet161_Modified(
            embedding_size=EMBEDDING_DIM,
            class_size=NUM_CLASSES,
            pretrained=False,
            only_embeddings=True,
            l2_normed=True
        )
        densenet.eval()
        models["EUSIPCO-DenseNet161"] = (EusipcoEvalWrapper(densenet), (1, 3, 228, 228))
        print("  [OK] EUSIPCO-DenseNet161")
    except Exception as e:
        print(f"  [SKIP] EUSIPCO-DenseNet161: {e}")

    # ------------------------------------------------------------------
    # 7-9. timm ViT Baselines (used as SCA backbones)
    # ------------------------------------------------------------------
    try:
        import timm

        for timm_name, display_name, input_h in [
            ("deit_tiny_patch16_224", "DeiT-Tiny", 224),
            ("mobilevit_s", "MobileViT-S", 256),
            ("swin_tiny_patch4_window7_224", "Swin-Tiny", 224),
        ]:
            m = timm.create_model(timm_name, pretrained=False, num_classes=0)
            m.eval()
            models[display_name] = (m, (1, 3, input_h, input_h))
            print(f"  [OK] {display_name}")
    except Exception as e:
        print(f"  [SKIP] timm baselines: {e}")

    # ------------------------------------------------------------------
    # 10-12. timm CNN Baselines
    # ------------------------------------------------------------------
    try:
        import timm

        for timm_name, display_name in [
            ("resnet50", "ResNet-50"),
            ("efficientnet_b0", "EfficientNet-B0"),
            ("mobilenetv3_small_100", "MobileNetV3-Small"),
        ]:
            m = timm.create_model(timm_name, pretrained=False, num_classes=0)
            m.eval()
            models[display_name] = (m, (1, 3, 224, 224))
            print(f"  [OK] {display_name}")
    except Exception as e:
        print(f"  [SKIP] timm CNN baselines: {e}")

    return models


# ============================================================
# RUN BENCHMARK
# ============================================================
def run_benchmark():
    print("=" * 80)
    print("INFERENCE BENCHMARK — All Models")
    print("=" * 80)
    print(f"Device GPU : {DEVICE_GPU}")
    print(f"CUDA       : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name   : {torch.cuda.get_device_name(0)}")
    print(f"Warmup runs: {N_WARMUP}")
    print(f"Timed runs : {N_RUNS}")
    print("=" * 80)

    print("\nLoading models...")
    models = build_models()
    results = {}

    print(f"\n{'=' * 80}")
    print(f"Running benchmarks on {len(models)} models...")
    print(f"{'=' * 80}")

    for name, (model, input_size) in models.items():
        print(f"\n>>> {name}  (input: {input_size})")
        total_params, trainable_params = count_params(model)

        # FLOPs
        flops = compute_flops(model, input_size)
        flops_str = f"{flops / 1e9:.2f} G" if flops else "N/A"

        # GPU benchmark
        if DEVICE_GPU.type == "cuda":
            try:
                gpu_result = measure_latency(model, DEVICE_GPU, input_size)
                print(f"    GPU: {gpu_result['mean_ms']:.2f} +/- {gpu_result['std_ms']:.2f} ms | "
                      f"FPS: {gpu_result['fps']:.1f}")
            except Exception as e:
                print(f"    GPU: FAILED ({e})")
                gpu_result = {"mean_ms": -1, "std_ms": -1, "fps": -1, "device": str(DEVICE_GPU)}
        else:
            gpu_result = {"mean_ms": -1, "std_ms": -1, "fps": -1, "device": "N/A (no CUDA)"}

        # CPU benchmark
        try:
            cpu_result = measure_latency(model, DEVICE_CPU, input_size)
            print(f"    CPU: {cpu_result['mean_ms']:.2f} +/- {cpu_result['std_ms']:.2f} ms | "
                  f"FPS: {cpu_result['fps']:.1f}")
        except Exception as e:
            print(f"    CPU: FAILED ({e})")
            cpu_result = {"mean_ms": -1, "std_ms": -1, "fps": -1, "device": "cpu"}

        print(f"    Params: {total_params / 1e6:.2f}M | FLOPs: {flops_str}")

        results[name] = {
            "params_total": total_params,
            "params_M": round(total_params / 1e6, 2),
            "flops": flops,
            "flops_G": round(flops / 1e9, 2) if flops else None,
            "input_size": list(input_size),
            "gpu": gpu_result,
            "cpu": cpu_result,
        }

    # Save JSON
    out_json = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_json}")

    # Print summary table
    print("\n" + "=" * 110)
    print(f"{'Model':<28} {'Params(M)':>10} {'FLOPs(G)':>10} "
          f"{'GPU(ms)':>14} {'CPU(ms)':>14} {'GPU FPS':>10}")
    print("-" * 110)

    for name, r in results.items():
        flops_g = f"{r['flops_G']:.2f}" if r['flops_G'] else "N/A"
        gpu_ms = (f"{r['gpu']['mean_ms']:.2f}+/-{r['gpu']['std_ms']:.2f}"
                  if r['gpu']['mean_ms'] > 0 else "N/A")
        cpu_ms = (f"{r['cpu']['mean_ms']:.2f}+/-{r['cpu']['std_ms']:.2f}"
                  if r['cpu']['mean_ms'] > 0 else "N/A")
        gpu_fps = f"{r['gpu']['fps']:.1f}" if r['gpu']['fps'] > 0 else "N/A"

        print(f"{name:<28} {r['params_M']:>10.2f} {flops_g:>10} "
              f"{gpu_ms:>14} {cpu_ms:>14} {gpu_fps:>10}")

    print("=" * 110)

    # Print LaTeX table for paper
    print("\n% LaTeX table (copy-paste into paper)")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Computational complexity comparison of all models.}")
    print("\\label{tab:inference_benchmark}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Model & Params (M) & FLOPs (G) & GPU (ms) & CPU (ms) \\\\")
    print("\\midrule")
    for name, r in results.items():
        flops_g = f"{r['flops_G']:.2f}" if r['flops_G'] else "--"
        gpu_ms = f"{r['gpu']['mean_ms']:.2f}" if r['gpu']['mean_ms'] > 0 else "--"
        cpu_ms = f"{r['cpu']['mean_ms']:.2f}" if r['cpu']['mean_ms'] > 0 else "--"
        latex_name = name.replace("_", "\\_")
        print(f"{latex_name} & {r['params_M']:.2f} & {flops_g} & {gpu_ms} & {cpu_ms} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    return results


if __name__ == "__main__":
    results = run_benchmark()
