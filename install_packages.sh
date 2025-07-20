#!/bin/bash
source /root/tf_gpu_venv/bin/activate
cd /mnt/d/Conference

# Install remaining packages
pip install opencv-python
pip install librosa
pip install matplotlib
pip install scikit-learn
pip install requests

# Try to install pyaudio (with system dependencies already installed)
pip install pyaudio

echo "Package installation completed!" 