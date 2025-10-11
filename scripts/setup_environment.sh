#!/bin/bash
# Setup script for Equation-CLIP project environment

set -e  # Exit on error

echo "======================================"
echo "Equation-CLIP Environment Setup"
echo "======================================"

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (adjust for your CUDA version)
echo "Installing PyTorch..."
echo "Detecting CUDA availability..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected. Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected. Installing CPU-only PyTorch..."
    pip install torch torchvision
fi

# Install PyTorch Geometric
echo "Installing PyTorch Geometric..."
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install other requirements
echo "Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "======================================"
echo "Environment setup complete!"
echo "======================================"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'"
echo ""
