#!/bin/bash

# SeqTrack Assignment 3 - WSL Setup Script
# Run this script in WSL to set up the environment

echo "=== SeqTrack Assignment 3 - WSL Setup ==="
echo "Setting up environment for PyCharm + WSL deployment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential build tools
echo "🔧 Installing build tools..."
sudo apt install -y build-essential cmake git curl wget

# Install Python and pip
echo "🐍 Installing Python and pip..."
sudo apt install -y python3 python3-pip python3-venv python3-dev

# Create virtual environment
echo "🏠 Creating virtual environment..."
python3 -m venv seqtrack_env
source seqtrack_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version for WSL)
echo "🔥 Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install numpy pandas matplotlib tqdm PyYAML easydict

# Install datasets and transformers
echo "🤗 Installing Hugging Face libraries..."
pip install datasets transformers

# Install SeqTrack dependencies
echo "🎯 Installing SeqTrack dependencies..."
pip install cython opencv-python pycocotools jpeg4py lmdb scipy visdom timm yacs

# Install additional useful packages
echo "🛠️ Installing additional packages..."
pip install colorama tb-nightly tikzplotlib

# Set up environment variables
echo "🔧 Setting up environment variables..."
export PYTHONPATH="/home/$USER/assignment_3:$PYTHONPATH"
export PYTHONPATH="/home/$USER/assignment_3/SeqTrack:$PYTHONPATH"
export PYTHONPATH="/home/$USER/assignment_3/SeqTrack/lib:$PYTHONPATH"

# Add to bashrc for persistence
echo 'export PYTHONPATH="/home/$USER/assignment_3:$PYTHONPATH"' >> ~/.bashrc
echo 'export PYTHONPATH="/home/$USER/assignment_3/SeqTrack:$PYTHONPATH"' >> ~/.bashrc
echo 'export PYTHONPATH="/home/$USER/assignment_3/SeqTrack/lib:$PYTHONPATH"' >> ~/.bashrc

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p assignment_3/checkpoints
mkdir -p assignment_3/logs

# Test installation
echo "🧪 Testing installation..."
python3 -c "import torch; print('✅ PyTorch version:', torch.__version__)"
python3 -c "import datasets; print('✅ Datasets library available')"
python3 -c "import numpy; print('✅ NumPy version:', numpy.__version__)"

# Set permissions
echo "🔐 Setting file permissions..."
chmod +x assignment_3/seqtrack_train.py
chmod +x assignment_3/dataset_loader.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Copy your assignment files to /home/$USER/assignment_3/"
echo "2. Copy SeqTrack repository to /home/$USER/assignment_3/SeqTrack/"
echo "3. Activate virtual environment: source seqtrack_env/bin/activate"
echo "4. Run training: python3 assignment_3/seqtrack_train.py"
echo ""
echo "🐧 For PyCharm integration:"
echo "1. Open PyCharm and configure WSL interpreter"
echo "2. Set project root to /home/$USER/assignment_3/"
echo "3. Use WSL terminal in PyCharm"
echo ""
echo "📖 See PYCHARM_WSL_DEPLOYMENT.md for detailed instructions"
