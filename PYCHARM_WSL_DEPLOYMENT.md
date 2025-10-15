# PyCharm + WSL Deployment Guide for SeqTrack Assignment 3

## 🐧 Prerequisites

### 1. WSL Setup
Ensure you have WSL2 installed with Ubuntu:
```bash
# Check WSL version
wsl --list --verbose

# If WSL1, upgrade to WSL2
wsl --set-version Ubuntu-20.04 2
```

### 2. PyCharm Configuration
- **PyCharm Professional** (recommended) or **PyCharm Community**
- **WSL Plugin** enabled
- **Python Plugin** enabled

## 🚀 Step-by-Step Deployment

### Step 1: Configure WSL in PyCharm

1. **Open PyCharm** and go to `File > Settings` (or `PyCharm > Preferences` on Mac)

2. **Navigate to Build, Execution, Deployment > Deployment**
   - Click the `+` button to add a new deployment configuration
   - Choose `WSL` as the deployment type
   - Name it: `WSL Ubuntu`

3. **Configure WSL Connection**
   - WSL distribution: `Ubuntu-20.04` (or your WSL version)
   - Leave other settings as default
   - Test connection to ensure it works

### Step 2: Set Up Python Interpreter

1. **Go to Project Settings > Python Interpreter**
2. **Click the gear icon > Add**
3. **Select WSL**
4. **Choose your WSL distribution**
5. **Select Python path** (usually `/usr/bin/python3` or `/home/username/.local/bin/python3`)

### Step 3: Transfer Project to WSL

#### Option A: Using PyCharm Deployment
1. **Right-click on `assignment_3` folder** in PyCharm
2. **Select "Upload to..." > WSL Ubuntu**
3. **Choose destination**: `/home/yourusername/assignment_3`

#### Option B: Manual Transfer
```bash
# In WSL terminal
cd /home/yourusername
mkdir -p assignment_3
cd assignment_3

# Copy files from Windows (adjust path as needed)
cp -r /mnt/d/study/level\ 4/Image_processing-1/Project/Assignment_3/assignment_3/* .
```

### Step 4: Install Dependencies in WSL

```bash
# Update package lists
sudo apt update

# Install Python and pip if not already installed
sudo apt install python3 python3-pip python3-venv

# Create virtual environment (recommended)
python3 -m venv seqtrack_env
source seqtrack_env/bin/activate

# Install PyTorch for CPU (WSL compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt

# Install additional WSL-specific packages
sudo apt install build-essential cmake
pip install cython
```

### Step 5: Configure PyCharm for WSL Development

1. **Set WSL as Default Interpreter**
   - Go to `File > Settings > Project > Python Interpreter`
   - Select the WSL Python interpreter you configured

2. **Configure Run/Debug Configuration**
   - Go to `Run > Edit Configurations`
   - Click `+` and select `Python`
   - Name: `SeqTrack Training`
   - Script path: `/home/yourusername/assignment_3/seqtrack_train.py`
   - Python interpreter: WSL interpreter
   - Working directory: `/home/yourusername/assignment_3`

### Step 6: Set Up SeqTrack Repository in WSL

```bash
# In WSL, navigate to your project directory
cd /home/yourusername/assignment_3

# Clone or copy SeqTrack repository
# Option 1: If you have it in Windows, copy it
cp -r /mnt/d/study/level\ 4/Image_processing-1/Project/Assignment_3/SeqTrack .

# Option 2: Clone fresh copy
git clone https://github.com/microsoft/VideoX.git
mv VideoX/master/SeqTrack .
```

### Step 7: Configure Environment Variables

```bash
# Add to ~/.bashrc or ~/.profile
export PYTHONPATH="/home/yourusername/assignment_3:$PYTHONPATH"
export PYTHONPATH="/home/yourusername/assignment_3/SeqTrack:$PYTHONPATH"
export PYTHONPATH="/home/yourusername/assignment_3/SeqTrack/lib:$PYTHONPATH"

# Reload shell
source ~/.bashrc
```

### Step 8: Test the Setup

1. **Open WSL Terminal in PyCharm**
   - Go to `Tools > Start SSH Session`
   - Select your WSL configuration

2. **Test Python Environment**
```bash
cd /home/yourusername/assignment_3
python3 -c "import torch; print('PyTorch version:', torch.__version__)"
python3 -c "import datasets; print('Datasets available')"
```

3. **Run Dataset Loader**
```bash
python3 dataset_loader.py
```

4. **Run Training Script**
```bash
python3 seqtrack_train.py
```

## 🔧 PyCharm-Specific Configuration

### Terminal Configuration
1. **Go to Settings > Tools > Terminal**
2. **Set Shell path**: `/bin/bash` or `/bin/zsh`
3. **Working directory**: Your project directory

### Code Style and Linting
1. **Install Python packages in WSL**
```bash
pip install flake8 black isort
```

2. **Configure in PyCharm**
   - Go to Settings > Tools > External Tools
   - Add flake8, black, and isort as external tools

### Version Control
```bash
# Initialize git in WSL
cd /home/yourusername/assignment_3
git init
git add .
git commit -m "Initial SeqTrack assignment setup"
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# If you get import errors, check Python path
echo $PYTHONPATH
python3 -c "import sys; print(sys.path)"

# Add SeqTrack to Python path
export PYTHONPATH="/home/yourusername/assignment_3/SeqTrack:$PYTHONPATH"
```

#### 2. CUDA/GPU Issues
```bash
# For WSL, use CPU-only PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 3. Permission Issues
```bash
# Fix file permissions
sudo chown -R $USER:$USER /home/yourusername/assignment_3
chmod +x seqtrack_train.py
```

#### 4. Memory Issues
```bash
# Monitor memory usage
free -h
htop

# If low memory, reduce batch size in training script
```

### WSL-Specific Optimizations

#### 1. Performance Tuning
```bash
# Create .wslconfig in Windows user directory
# C:\Users\YourUsername\.wslconfig
[wsl2]
memory=8GB
processors=4
swap=2GB
```

#### 2. File System Performance
```bash
# Work in WSL file system, not Windows mounted drives
cd /home/yourusername/assignment_3
# NOT /mnt/d/...
```

## 🎯 Running the Assignment

### Method 1: PyCharm Run Configuration
1. **Select the run configuration** you created
2. **Click the Run button** (green arrow)
3. **Monitor output** in PyCharm's Run window

### Method 2: WSL Terminal
```bash
# Activate virtual environment
source seqtrack_env/bin/activate

# Run training
cd /home/yourusername/assignment_3
python3 seqtrack_train.py
```

### Method 3: Debug Mode
1. **Set breakpoints** in your code
2. **Select Debug configuration**
3. **Click Debug button** (bug icon)
4. **Step through code** using PyCharm's debugger

## 📊 Expected Output

When running successfully, you should see:
```
=== Assignment 3: SeqTrack Setup, Training, and Checkpoint Management ===
Team Number: 8
Seed: 8, Epochs: 5, Patch Size: 1
======================================================================

Random seed set to 8
Loading LaSOT dataset from Hugging Face...
Selected classes: ['airplane', 'bicycle']
Starting epoch 1/5
Epoch 1: 0/270 | Loss: 0.8234 | IoU: 0.7234 | Time for last 50 samples: 0:00:15 | Time since beginning: 0:00:15 | Time left to finish epoch: 0:01:30
...
✅ Training completed successfully.
Checkpoints saved in: assignment_3/checkpoints/
Log file: assignment_3/training_log.txt
```

## 🔍 Monitoring and Debugging

### PyCharm Tools
- **Run Window**: Shows real-time output
- **Debug Console**: Interactive debugging
- **Terminal**: Direct WSL access
- **File Watcher**: Automatic file synchronization

### WSL Monitoring
```bash
# Monitor system resources
htop
nvidia-smi  # if you have GPU access
free -h

# Monitor training progress
tail -f training_log.txt
```

## 📝 Best Practices

1. **Always work in WSL file system** for better performance
2. **Use virtual environments** to isolate dependencies
3. **Set up proper Python path** for SeqTrack imports
4. **Use PyCharm's integrated terminal** for seamless workflow
5. **Enable auto-save** and **file watchers** for smooth development
6. **Use version control** to track changes

## 🎉 Success Checklist

- [ ] WSL2 installed and configured
- [ ] PyCharm connected to WSL
- [ ] Python interpreter set to WSL
- [ ] Dependencies installed in WSL
- [ ] SeqTrack repository accessible
- [ ] Environment variables configured
- [ ] Training script runs without errors
- [ ] Checkpoints saved successfully
- [ ] Logs generated properly

Your SeqTrack Assignment 3 is now ready to run in PyCharm with WSL! 🚀
