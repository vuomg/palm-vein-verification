@echo off
REM setup.bat - Quick setup script for Palm Vein Recognition platform (Windows)

echo ================================
echo Palm Vein Setup
echo ================================
echo.

REM 1. Create virtual environment
echo [1/4] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM 2. Upgrade pip
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM 3. Install PyTorch (CPU version - change to CUDA if needed)
echo [3/4] Installing PyTorch...
pip install torch torchvision torchaudio

REM 4. Install project requirements
echo [4/4] Installing project requirements...
pip install -r requirements.txt

echo.
echo ================================
echo Installation Complete!
echo ================================
echo.
echo Activate environment with:
echo   venv\Scripts\activate.bat
echo.
echo Prepare dataset with:
echo   python preprocessing/prepare_scut_dataset.py
echo.
echo Split dataset with:
echo   python preprocessing/split_scut_openset.py
echo.
echo Run training with:
echo   run_scut_baselines.bat
echo.
pause
