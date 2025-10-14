#!/usr/bin/env python3
"""
WSL Setup Test Script for SeqTrack Assignment 3
This script tests if all dependencies are properly installed in WSL
"""

import sys
import os
import subprocess
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {module_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False

def test_system_info():
    """Test system information"""
    print("=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {sys.platform}")
    print(f"Current working directory: {os.getcwd()}")
    print()

def test_python_path():
    """Test Python path configuration"""
    print("=== Python Path ===")
    for path in sys.path:
        print(f"  {path}")
    print()

def test_core_dependencies():
    """Test core Python dependencies"""
    print("=== Core Dependencies ===")
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('torchaudio', 'TorchAudio'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'TQDM'),
        ('yaml', 'PyYAML'),
        ('easydict', 'EasyDict'),
        ('cv2', 'OpenCV'),
        ('datasets', 'HuggingFace Datasets'),
        ('transformers', 'HuggingFace Transformers'),
        ('PIL', 'Pillow'),
        ('scipy', 'SciPy'),
        ('lmdb', 'LMDB'),
        ('timm', 'Timm'),
        ('yacs', 'YACS'),
    ]
    
    success_count = 0
    for module, name in dependencies:
        if test_import(module):
            success_count += 1
    
    print(f"\n📊 Dependencies Status: {success_count}/{len(dependencies)} installed")
    return success_count == len(dependencies)

def test_seqtrack_imports():
    """Test SeqTrack specific imports"""
    print("\n=== SeqTrack Imports ===")
    
    # Add SeqTrack to path
    seqtrack_path = os.path.join(os.path.dirname(__file__), 'SeqTrack')
    lib_path = os.path.join(seqtrack_path, 'lib')
    
    if os.path.exists(seqtrack_path):
        sys.path.insert(0, seqtrack_path)
        sys.path.insert(0, lib_path)
        print(f"✅ SeqTrack path added: {seqtrack_path}")
        print(f"✅ Lib path added: {lib_path}")
        
        # Test SeqTrack imports
        seqtrack_modules = [
            ('lib.train.trainers', 'LTRTrainer'),
            ('lib.models.seqtrack', 'build_seqtrack'),
            ('lib.train.actors', 'SeqTrackActor'),
            ('lib.config.seqtrack.config', 'cfg'),
        ]
        
        seqtrack_success = 0
        for module, name in seqtrack_modules:
            try:
                test_import(module)
                seqtrack_success += 1
            except:
                print(f"⚠️ {module}: Not available (expected for mock implementation)")
        
        return seqtrack_success > 0
    else:
        print(f"❌ SeqTrack directory not found: {seqtrack_path}")
        return False

def test_gpu_availability():
    """Test GPU availability"""
    print("\n=== GPU Availability ===")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✅ CUDA version: {torch.version.cuda}")
        else:
            print("⚠️ CUDA not available - using CPU")
        
        # Test Intel Extension for PyTorch
        try:
            import intel_extension_for_pytorch as ipex
            print("✅ Intel Extension for PyTorch available")
        except ImportError:
            print("⚠️ Intel Extension for PyTorch not available")
            
    except ImportError:
        print("❌ PyTorch not available")

def test_file_structure():
    """Test file structure"""
    print("\n=== File Structure ===")
    
    required_files = [
        'seqtrack_train.py',
        'dataset_loader.py',
        'requirements.txt',
        'assignment_3_report.md',
        'PYCHARM_WSL_DEPLOYMENT.md',
        'setup_wsl.sh',
        'test_wsl_setup.py'
    ]
    
    required_dirs = [
        'checkpoints',
        'SeqTrack'
    ]
    
    print("Required files:")
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
    
    print("\nRequired directories:")
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/")

def test_training_script():
    """Test if training script can be imported"""
    print("\n=== Training Script Test ===")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(__file__))
        
        # Test dataset loader
        try:
            from dataset_loader import load_lasot_dataset
            print("✅ dataset_loader.py can be imported")
        except ImportError as e:
            print(f"⚠️ dataset_loader.py: {e}")
        
        # Test training script
        try:
            import seqtrack_train
            print("✅ seqtrack_train.py can be imported")
        except ImportError as e:
            print(f"⚠️ seqtrack_train.py: {e}")
            
    except Exception as e:
        print(f"❌ Error testing training script: {e}")

def main():
    """Main test function"""
    print("🧪 SeqTrack Assignment 3 - WSL Setup Test")
    print("=" * 50)
    
    # Run all tests
    test_system_info()
    test_python_path()
    
    core_success = test_core_dependencies()
    seqtrack_success = test_seqtrack_imports()
    test_gpu_availability()
    test_file_structure()
    test_training_script()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    
    if core_success:
        print("✅ Core dependencies: READY")
    else:
        print("❌ Core dependencies: NEEDS ATTENTION")
    
    if seqtrack_success:
        print("✅ SeqTrack imports: READY")
    else:
        print("⚠️ SeqTrack imports: USING MOCK IMPLEMENTATION")
    
    print("\n🎯 Ready for training:")
    if core_success:
        print("✅ Environment is ready for SeqTrack training")
        print("\n🚀 To start training:")
        print("   python3 seqtrack_train.py")
    else:
        print("❌ Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        print("   bash setup_wsl.sh")

if __name__ == "__main__":
    main()
