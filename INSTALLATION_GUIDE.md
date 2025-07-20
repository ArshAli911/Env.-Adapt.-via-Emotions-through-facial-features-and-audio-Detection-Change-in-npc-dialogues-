# Installation Guide

## VR Emotion Adaptation System - Complete Setup Guide

This guide provides step-by-step instructions for setting up the VR Emotion Adaptation System on different platforms.

## 📋 Prerequisites

### Hardware Requirements

#### Minimum Requirements
- **CPU**: Intel i5-8400 / AMD Ryzen 5 2600 or equivalent
- **RAM**: 8GB DDR4
- **Storage**: 10GB free space
- **GPU**: Integrated graphics (CPU mode)
- **Camera**: Any USB webcam (720p minimum)
- **Microphone**: Built-in or USB microphone

#### Recommended Requirements
- **CPU**: Intel i7-10700K / AMD Ryzen 7 3700X or better
- **RAM**: 16GB DDR4 or higher
- **Storage**: 20GB free space (SSD recommended)
- **GPU**: NVIDIA GTX 1060 / RTX 3060 or better (CUDA support)
- **Camera**: 1080p webcam with good low-light performance
- **Microphone**: Dedicated USB microphone or headset

### Software Requirements
- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS 12+
- **Python**: 3.8 - 3.11 (3.11 recommended)
- **Git**: Latest version
- **CUDA**: 11.8+ (for GPU acceleration)

## 🖥️ Platform-Specific Installation

### Windows Installation

#### Step 1: Install Python and Git
```powershell
# Download and install Python 3.11 from python.org
# Make sure to check "Add Python to PATH"

# Install Git from git-scm.com
# Or use winget
winget install Git.Git
winget install Python.Python.3.11
```

#### Step 2: Install CUDA (Optional but Recommended)
```powershell
# Download CUDA Toolkit 11.8+ from NVIDIA website
# https://developer.nvidia.com/cuda-downloads

# Verify installation
nvidia-smi
nvcc --version
```

#### Step 3: Clone Repository
```powershell
git clone https://github.com/your-username/vr-emotion-adaptation.git
cd vr-emotion-adaptation
```

#### Step 4: Create Virtual Environment
```powershell
# Create virtual environment
python -m venv venv_py311_local

# Activate virtual environment
.\venv_py311_local\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

#### Step 5: Install Dependencies
```powershell
# Install PyTorch with CUDA support (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Install additional Windows-specific packages
pip install pywin32
```

### Linux (Ubuntu) Installation

#### Step 1: Update System and Install Dependencies
```bash
sudo apt update && sudo apt upgrade -y

# Install Python and development tools
sudo apt install -y python3.11 python3.11-venv python3.11-dev
sudo apt install -y build-essential cmake git

# Install audio dependencies
sudo apt install -y portaudio19-dev libasound2-dev

# Install OpenCV dependencies
sudo apt install -y libopencv-dev python3-opencv

# Install system libraries for GUI
sudo apt install -y libgtk-3-dev libglib2.0-dev
```

#### Step 2: Install CUDA (Optional)
```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA
sudo apt-get -y install cuda

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Step 3: Clone and Setup Project
```bash
git clone https://github.com/your-username/vr-emotion-adaptation.git
cd vr-emotion-adaptation

# Create virtual environment
python3.11 -m venv venv_py311_local

# Activate virtual environment
source venv_py311_local/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### Step 4: Install Python Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

### macOS Installation

#### Step 1: Install Homebrew and Dependencies
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and dependencies
brew install python@3.11 git cmake portaudio

# Install OpenCV
brew install opencv
```

#### Step 2: Clone and Setup Project
```bash
git clone https://github.com/your-username/vr-emotion-adaptation.git
cd vr-emotion-adaptation

# Create virtual environment
python3.11 -m venv venv_py311_local

# Activate virtual environment
source venv_py311_local/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### Step 3: Install Dependencies
```bash
# Install PyTorch (CPU version for macOS)
pip install torch torchvision torchaudio

# Install other requirements
pip install -r requirements.txt
```

## 🤖 Ollama Installation and Setup

### Install Ollama

#### Windows
```powershell
# Download from https://ollama.ai/download
# Or use winget
winget install Ollama.Ollama
```

#### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### macOS
```bash
brew install ollama
```

### Setup Ollama Models

```bash
# Start Ollama service (Linux/macOS)
ollama serve

# Windows: Ollama starts automatically after installation

# Pull recommended models
ollama pull deepseek-r1:latest
ollama pull qwen2.5:latest

# Verify installation
ollama list
```

### Configure Ollama for the Project

```bash
# Test Ollama connectivity
python test_ollama.py

# Expected output:
# ✅ Ollama package imported successfully
# ✅ Successfully connected to Ollama server
# ✅ Found 2 usable model(s): ['deepseek-r1:latest', 'qwen2.5:latest']
# ✅ Dialogue generation successful!
```

## 🔧 Configuration and Testing

### Verify Installation

#### Test Python Environment
```bash
# Activate virtual environment
source venv_py311_local/bin/activate  # Linux/macOS
# or
.\venv_py311_local\Scripts\Activate.ps1  # Windows

# Test Python packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
```

#### Test Hardware Access
```bash
# Test camera access
python test_webcam.py

# Test audio access
python test_audio.py

# Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Configure System Settings

#### Edit Configuration Files

**config.py** - Core system settings:
```python
# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Camera settings
CAMERA_INDEX = 0  # Change if you have multiple cameras
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# Processing settings
TARGET_FPS = 10
CONFIDENCE_THRESHOLD = 0.6

# Enable/disable features
ENABLE_AUDIO = True
ENABLE_FACIAL = True
ENABLE_OLLAMA = True
```

**config_unified.py** - VR adaptation settings:
```python
# Emotion adaptation parameters
EMOTION_ADAPTATION_STRENGTH = 1.0
TEMPORAL_SMOOTHING_WINDOW = 5
DIALOGUE_GENERATION_THRESHOLD = 0.6
```

### First Run

```bash
# Activate virtual environment
source venv_py311_local/bin/activate  # Linux/macOS
# or
.\venv_py311_local\Scripts\Activate.ps1  # Windows

# Launch the application
streamlit run app.py
```

The application should open in your browser at `http://localhost:8501`.

## 🚨 Troubleshooting Common Issues

### Python Environment Issues

#### Issue: "Python not found"
```bash
# Windows
py -3.11 --version

# Linux/macOS
python3.11 --version

# If not found, reinstall Python 3.11
```

#### Issue: "pip not found"
```bash
# Reinstall pip
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

### Package Installation Issues

#### Issue: "Failed building wheel for [package]"
```bash
# Install build tools
# Windows
pip install wheel setuptools

# Linux
sudo apt install build-essential python3.11-dev

# macOS
xcode-select --install
```

#### Issue: "CUDA out of memory"
```python
# In config.py, reduce batch size or switch to CPU
DEVICE = 'cpu'  # Force CPU usage
BATCH_SIZE = 1  # Reduce batch size
```

### Camera and Audio Issues

#### Issue: "Camera not accessible"
```bash
# Check camera permissions
# Linux
sudo usermod -a -G video $USER
# Logout and login again

# Test different camera indices
python -c "import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).read()[0]}') for i in range(5)]"
```

#### Issue: "Audio device not found"
```bash
# Linux - install additional audio packages
sudo apt install pulseaudio-utils alsa-utils

# Test audio devices
python -c "import pyaudio; p=pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count())]"
```

### Ollama Issues

#### Issue: "Ollama connection failed"
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
# Linux/macOS
ollama serve

# Windows - restart Ollama service
net stop ollama
net start ollama
```

#### Issue: "Model not found"
```bash
# List available models
ollama list

# Pull missing models
ollama pull deepseek-r1:latest

# Check model size and available disk space
du -sh ~/.ollama/models/
```

### Performance Issues

#### Issue: "Low FPS (< 5 FPS)"
```python
# In config.py, optimize settings
TARGET_FPS = 5  # Reduce target FPS
VIDEO_WIDTH = 320  # Reduce resolution
VIDEO_HEIGHT = 240
ENABLE_TEMPORAL_SMOOTHING = False  # Disable smoothing
```

#### Issue: "High memory usage"
```python
# In config.py
CLEAR_CACHE_INTERVAL = 100  # Clear cache more frequently
MAX_HISTORY_LENGTH = 50     # Reduce history length
```

## 🔄 Updates and Maintenance

### Updating the System
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Update Ollama models
ollama pull deepseek-r1:latest
```

### Backup Configuration
```bash
# Backup your custom configurations
cp config.py config_backup.py
cp config_unified.py config_unified_backup.py
```

### Performance Monitoring
```bash
# Monitor system resources
# Linux
htop
nvidia-smi  # For GPU monitoring

# Windows
taskmgr
nvidia-smi
```

## 📞 Getting Help

If you encounter issues not covered in this guide:

1. **Check the logs**: Look in `logs/backend.log` for detailed error messages
2. **Run diagnostics**: Use `python diagnostic_test.py` for system health check
3. **GitHub Issues**: Report bugs and request features on the project repository
4. **Community**: Join our Discord/Slack for community support

## 🎯 Next Steps

After successful installation:

1. **Read the API Documentation**: `API_DOCUMENTATION.md`
2. **Explore Examples**: Check the `examples/` directory
3. **Customize Configuration**: Adapt settings for your specific use case
4. **Train Custom Models**: Use your own datasets with `train_models.py`
5. **Integrate with VR**: Follow VR integration guides for Unity/Unreal

Congratulations! Your VR Emotion Adaptation System is now ready to use. 🎉