#!/bin/bash
# Setup conda environment for 2D Gaussian Splatting project
# Usage: bash setup_env.sh
# No sudo required

set -e  # Exit on error

ENV_NAME="2dgs"
PYTHON_VERSION="3.11"

echo "========================================"
echo "  2D Gaussian Splatting Environment Setup"
echo "========================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda first."
    echo "  curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# Check CUDA version
echo ""
echo "[1/5] Checking CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo "  Found CUDA $CUDA_VERSION"
else
    echo "  WARNING: nvcc not found. Will install CPU-only PyTorch."
    CUDA_VERSION="cpu"
fi

# Create conda environment
echo ""
echo "[2/5] Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "  Environment '$ENV_NAME' already exists. Removing..."
    conda env remove -n $ENV_NAME -y
fi
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate environment
echo ""
echo "[3/5] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install PyTorch with appropriate CUDA version
echo ""
echo "[4/5] Installing PyTorch..."
if [[ "$CUDA_VERSION" == "cpu" ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
elif [[ "$CUDA_VERSION" == "12."* ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11.8"* ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "  CUDA $CUDA_VERSION detected. Installing default PyTorch (cu121)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

# Install other dependencies
echo ""
echo "[5/5] Installing dependencies..."
pip install \
    numpy \
    pillow>=9.0.0 \
    pyyaml>=6.0 \
    tqdm>=4.65.0 \
    matplotlib>=3.5.0 \
    pytorch-msssim>=1.0.0 \
    lpips>=0.1.4 \
    torchsort

# Install gsplat (CUDA required)
echo ""
echo "[Extra] Installing gsplat..."
if [[ "$CUDA_VERSION" != "cpu" ]]; then
    pip install gsplat
    echo "  gsplat installed successfully."
else
    echo "  WARNING: gsplat requires CUDA. Skipping."
    echo "  You can use the fallback CPU renderer (slower)."
fi

# Verify installation
echo ""
echo "========================================"
echo "  Verifying installation..."
echo "========================================"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')

import torchsort
print(f'  torchsort: OK')

try:
    from gsplat import rasterization
    print(f'  gsplat: OK')
except ImportError:
    print(f'  gsplat: Not installed (CPU mode)')

from pytorch_msssim import ssim
print(f'  pytorch-msssim: OK')

import lpips
print(f'  lpips: OK')
"

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run training:"
echo "  python scripts/train.py --config configs/config.yaml"
echo ""
