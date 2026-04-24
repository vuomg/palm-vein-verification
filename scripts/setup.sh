#!/bin/bash
# setup.sh - Quick setup script for Palm Vein Recognition platform

echo "================================"
echo "Palm Vein Setup"
echo "================================"
echo

# 1. Update system packages
echo "[1/5] Updating system packages..."
sudo apt update
sudo apt install -y python3 python3-pip python3-venv build-essential

# 2. Create virtual environment
echo "[2/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip
echo "[3/5] Upgrading pip..."
pip3 install --upgrade pip setuptools wheel

# 4. Install PyTorch (CPU version - change to CUDA if needed)
echo "[4/5] Installing PyTorch..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 5. Install project requirements
echo "[5/5] Installing project requirements..."
pip3 install -r requirements.txt

echo
echo "================================"
echo "Installation Complete!"
echo "================================"
echo
echo "Activate environment with:"
echo "  source venv/bin/activate"
echo
echo "Run training with:"
echo "  ./run_scut_baselines.sh"
echo
