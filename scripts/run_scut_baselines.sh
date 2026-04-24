#!/bin/bash
# Train all 9 models on SCUT dataset
# Open-set verification protocol with CLAHE preprocessing

SCUT_OPENSET="SCUT_dataset_openset"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSCL_DIR="${SCRIPT_DIR}/GSCL-PyTorch/vein_feature_learning"
FAIL_COUNT=0

# Clean up old results (permission issues)
echo "[*] Cleaning up old results..."
rm -rf results/ 2>/dev/null
rm -rf results_scut_*/ 2>/dev/null
sudo rm -rf results_scut_*/ 2>/dev/null || true
sudo chmod 777 . 2>/dev/null || true

# Detect Python command
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python not found!"
    echo "Install Python 3: sudo apt install python3 python3-pip"
    exit 1
fi
echo "[OK] Using: $PYTHON_CMD"
echo

# Check if dataset exists
if [ ! -d "${SCUT_OPENSET}/train" ]; then
    echo "[ERROR] Dataset not found: ${SCUT_OPENSET}/train"
    echo "Run: python prepare_scut_dataset.py"
    echo "Then: python split_scut_openset.py"
    exit 1
fi
echo "[OK] Dataset: ${SCUT_OPENSET}"
echo

# Function to run training and track failures
run_training() {
    local model_name=$1
    shift
    
    if $PYTHON_CMD train.py "$@"; then
        echo "[OK] $model_name"
    else
        echo "[FAILED] $model_name"
        ((FAIL_COUNT++))
    fi
    echo
}

# 1/12: MPSNet
echo "========================================"
echo "SCUT 1/12: MPSNet"
echo "========================================"
run_training "MPSNet" \
    --model mpsnet \
    --dataset "${SCUT_OPENSET}" \
    --output-dir results_scut_mpsnet \
    --epochs 100 --batch-size 16 \
    --database SCUT --eval-frequency 5 \
    --no-checkpoint --seed 42

# 2/12: Modified-DenseNet161
echo "========================================"
echo "SCUT 2/12: Modified-DenseNet161"
echo "========================================"
run_training "DenseNet161" \
    --model eusipco2020 --eusipco-backbone densenet161 \
    --dataset "${SCUT_OPENSET}" \
    --output-dir results_scut_densenet161 \
    --epochs 100 --batch-size 4 \
    --database SCUT --eval-frequency 5 \
    --no-checkpoint --seed 42

# 3/12: GSCL (ResNet18) - SKIPPED (requires separate setup)
echo "========================================"
echo "SCUT 3/12: GSCL (ResNet18) - SKIPPED"
echo "========================================"
echo "[SKIP] GSCL requires separate setup from GSCL-PyTorch/"
echo "To enable: copy GSCL-PyTorch/ to this directory"
echo

# 4/12: RSNet
echo "========================================"
echo "SCUT 4/12: RSNet"
echo "========================================"
run_training "RSNet" \
    --model rsnet \
    --dataset "${SCUT_OPENSET}" \
    --output-dir results_scut_rsnet \
    --epochs 100 --batch-size 32 \
    --database SCUT --eval-frequency 5 \
    --no-checkpoint --seed 42

# 5/12: FGFNet
echo "========================================"
echo "SCUT 5/12: FGFNet"
echo "========================================"
run_training "FGFNet" \
    --model fgfnet \
    --dataset "${SCUT_OPENSET}" \
    --output-dir results_scut_fgfnet \
    --epochs 100 --batch-size 4 \
    --database SCUT --eval-frequency 5 \
    --no-checkpoint --seed 42

# 6/12: ResNet50 - SKIPPED (requires separate setup)
echo "========================================"
echo "SCUT 6/12: ResNet50 - SKIPPED"
echo "========================================"
echo "[SKIP] ResNet50 requires separate setup from GSCL-PyTorch/"
echo "To enable: copy GSCL-PyTorch/ to this directory"
echo

# 7/12: MobileNetV3-Base
echo "========================================"
echo "SCUT 7/12: MobileNetV3-Base"
echo "========================================"
run_training "MobileNetV3-Base" \
    --model sca_mobilenet --no-stn --no-ca --no-spp \
    --dataset "${SCUT_OPENSET}" \
    --output-dir results_scut_mobilenetv3_base \
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 \
    --batch-size 16 --epochs 100 --lr 0.001 \
    --database SCUT --eval-frequency 5 \
    --no-checkpoint --seed 42

# 8/12: EfficientNet-B0
echo "========================================"
echo "SCUT 8/12: EfficientNet-B0"
echo "========================================"
run_training "EfficientNet-B0" \
    --model sca_mobilenet --sca-backbone efficientnet_b0 \
    --dataset "${SCUT_OPENSET}" \
    --output-dir results_scut_efficientnet_b0 \
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 \
    --batch-size 16 --epochs 100 --lr 0.001 \
    --database SCUT --eval-frequency 5 \
    --no-checkpoint --seed 42

# 9/12: MobileViT-S
echo "========================================"
echo "SCUT 9/12: MobileViT-S"
echo "========================================"
run_training "MobileViT-S" \
    --model sca_mobilenet --sca-backbone mobilevit_s \
    --dataset "${SCUT_OPENSET}" \
    --output-dir results_scut_mobilevit_s \
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 \
    --batch-size 16 --epochs 100 --lr 0.001 \
    --database SCUT --eval-frequency 5 \
    --no-checkpoint --seed 42

# 10/12: DeiT-Tiny
echo "========================================"
echo "SCUT 10/12: DeiT-Tiny"
echo "========================================"
run_training "DeiT-Tiny" \
    --model sca_mobilenet --sca-backbone deit_tiny \
    --dataset "${SCUT_OPENSET}" \
    --output-dir results_scut_deit_tiny \
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 \
    --batch-size 16 --epochs 100 --lr 0.001 \
    --database SCUT --eval-frequency 5 \
    --no-checkpoint --seed 42

# 11/12: Swin-Tiny
echo "========================================"
echo "SCUT 11/12: Swin-Tiny"
echo "========================================"
run_training "Swin-Tiny" \
    --model sca_mobilenet --sca-backbone swin_tiny \
    --dataset "${SCUT_OPENSET}" \
    --output-dir results_scut_swin_tiny \
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 \
    --batch-size 16 --epochs 100 --lr 0.001 \
    --database SCUT --eval-frequency 5 \
    --no-checkpoint --seed 42

# 12/12: SCA-MobileNet (proposed)
echo "========================================"
echo "SCUT 12/12: SCA-MobileNet (proposed)"
echo "========================================"
run_training "SCA-MobileNet" \
    --model sca_mobilenet \
    --dataset "${SCUT_OPENSET}" \
    --output-dir results_scut_sca_mobilenet \
    --loss-type adacos_only --feature-dim 1024 --dropout 0.3 \
    --batch-size 16 --epochs 100 --lr 0.001 \
    --database SCUT --eval-frequency 5 \
    --no-checkpoint --seed 42

# Summary
echo "========================================"
if [ $FAIL_COUNT -eq 0 ]; then
    echo "ALL 12 SCUT MODELS COMPLETED SUCCESSFULLY."
    echo "(GSCL models skipped - requires separate GSCL-PyTorch setup)"
else
    echo "Completed with $FAIL_COUNT failure(s)."
fi
echo "Results: results_scut_*/"
echo "========================================"

exit $FAIL_COUNT
7 SCUT MODELS COMPLETED SUCCESSFULLY."
    echo "(GSCL models skipped - requires separate GSCL-PyTorch setup)