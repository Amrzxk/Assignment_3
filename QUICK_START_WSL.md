# 🚀 Quick Start: PyCharm + WSL Deployment

## ⚡ Fast Setup (5 minutes)

### 1. Prerequisites Check
```bash
# Check if WSL is installed
wsl --list

# If not installed, install WSL2:
# Windows Store → Search "Ubuntu" → Install Ubuntu 20.04 LTS
```

### 2. Transfer Files to WSL
```bash
# From Windows PowerShell (run as administrator):
cd "D:\study\level 4\Image_processing-1\Project\Assignment_3"
.\assignment_3\transfer_to_wsl.bat
```

### 3. Setup WSL Environment
```bash
# Open WSL terminal:
wsl

# Navigate to project:
cd ~/assignment_3

# Run setup script:
bash setup_wsl.sh

# Activate environment:
source seqtrack_env/bin/activate
```

### 4. Test Setup
```bash
# Test all dependencies:
python3 test_wsl_setup.py

# Run training:
python3 seqtrack_train.py
```

### 5. Configure PyCharm
1. **Open PyCharm**
2. **File → Settings → Project → Python Interpreter**
3. **Add Interpreter → WSL**
4. **Select Ubuntu distribution**
5. **Set interpreter path**: `/home/yourusername/assignment_3/seqtrack_env/bin/python3`

## 🎯 Expected Results

After successful setup, you should see:
```
✅ Training completed successfully.
Checkpoints saved in: assignment_3/checkpoints/
Log file: assignment_3/training_log.txt
```

## 📁 Final Project Structure in WSL
```
/home/yourusername/assignment_3/
├── seqtrack_train.py          # Main training script
├── dataset_loader.py          # Dataset utilities
├── requirements.txt           # Dependencies
├── setup_wsl.sh              # WSL setup script
├── test_wsl_setup.py         # Test script
├── PYCHARM_WSL_DEPLOYMENT.md # Detailed guide
├── assignment_3_report.md    # Full report
├── checkpoints/              # Model checkpoints
└── SeqTrack/                 # SeqTrack repository
```

## 🔧 Troubleshooting

### Common Issues:
1. **Import errors**: Run `bash setup_wsl.sh` again
2. **Permission denied**: `chmod +x *.sh *.py`
3. **Path issues**: Check PYTHONPATH in ~/.bashrc
4. **PyCharm connection**: Restart PyCharm after WSL setup

### Quick Fixes:
```bash
# Fix permissions:
chmod +x *.sh *.py

# Reinstall dependencies:
pip install -r requirements.txt

# Test Python path:
python3 -c "import sys; print(sys.path)"
```

## 📖 Full Documentation

For detailed instructions, see:
- `PYCHARM_WSL_DEPLOYMENT.md` - Complete deployment guide
- `assignment_3_report.md` - Full technical documentation

## 🎉 Success!

Your SeqTrack Assignment 3 is now ready to run in PyCharm with WSL! 🚀
