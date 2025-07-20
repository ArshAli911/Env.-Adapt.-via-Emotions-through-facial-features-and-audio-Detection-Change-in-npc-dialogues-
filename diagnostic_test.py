#!/usr/bin/env python3
"""
Diagnostic script to identify current issues with the VR Emotion Adaptation app
"""

import sys
import traceback

def test_imports():
    """Test all required imports"""
    print("=== Testing Imports ===")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except Exception as e:
        print(f"❌ Streamlit import failed: {e}")
        
    try:
        import torch
        print(f"✅ PyTorch imported successfully (version: {torch.__version__})")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        
    try:
        import cv2
        print(f"✅ OpenCV imported successfully (version: {cv2.__version__})")
    except Exception as e:
        print(f"❌ OpenCV import failed: {e}")
        
    try:
        import pyaudio
        print("✅ PyAudio imported successfully")
    except Exception as e:
        print(f"❌ PyAudio import failed: {e}")
        
    try:
        import librosa
        print(f"✅ Librosa imported successfully (version: {librosa.__version__})")
    except Exception as e:
        print(f"❌ Librosa import failed: {e}")
        
    try:
        from streamlit_autorefresh import st_autorefresh
        print("✅ streamlit_autorefresh imported successfully")
    except Exception as e:
        print(f"❌ streamlit_autorefresh import failed: {e}")

def test_custom_modules():
    """Test custom module imports"""
    print("\n=== Testing Custom Modules ===")
    
    try:
        from data.data_loader import FacialExpressionDataset, AudioEmotionDataset
        print("✅ Data loaders imported successfully")
    except Exception as e:
        print(f"❌ Data loaders import failed: {e}")
        traceback.print_exc()
        
    try:
        from models.emotion_models import FacialEmotionCNN, AudioEmotionLSTM, MultiModalEmotionFusion, EmotionClassifier
        print("✅ Emotion models imported successfully")
    except Exception as e:
        print(f"❌ Emotion models import failed: {e}")
        traceback.print_exc()
        
    try:
        from vr_adaptation.environment_controller import EnvironmentController, EmotionType
        print("✅ Environment controller imported successfully")
    except Exception as e:
        print(f"❌ Environment controller import failed: {e}")
        traceback.print_exc()
        
    try:
        from utils import check_ollama_and_model
        print("✅ Utils imported successfully")
    except Exception as e:
        print(f"❌ Utils import failed: {e}")
        traceback.print_exc()
        
    try:
        from vr_emotion_adaptation import VREmotionAdaptation
        print("✅ VREmotionAdaptation imported successfully")
    except Exception as e:
        print(f"❌ VREmotionAdaptation import failed: {e}")
        traceback.print_exc()

def test_hardware():
    """Test hardware availability"""
    print("\n=== Testing Hardware ===")
    
    # Test webcam
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Webcam working (frame shape: {frame.shape})")
            else:
                print("❌ Webcam opened but can't read frames")
            cap.release()
        else:
            print("❌ Webcam not available")
    except Exception as e:
        print(f"❌ Webcam test failed: {e}")
    
    # Test audio
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        input_devices = []
        for i in range(numdevices):
            device_info = p.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:
                input_devices.append(device_info.get('name'))
        
        if input_devices:
            print(f"✅ Audio input devices found: {len(input_devices)}")
            for device in input_devices[:3]:  # Show first 3
                print(f"   - {device}")
        else:
            print("❌ No audio input devices found")
        p.terminate()
    except Exception as e:
        print(f"❌ Audio test failed: {e}")

def test_ollama():
    """Test Ollama connection"""
    print("\n=== Testing Ollama ===")
    
    try:
        from utils import check_ollama_and_model
        ollama_ok, ollama_msg = check_ollama_and_model("deepseek-r1:latest")
        if ollama_ok:
            print("✅ Ollama connection successful")
        else:
            print(f"❌ Ollama connection failed: {ollama_msg}")
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")

def test_model_files():
    """Test model file availability"""
    print("\n=== Testing Model Files ===")
    
    import os
    model_files = [
        "models/facial_emotion_cnn.pth",
        "models/audio_emotion_lstm.pth", 
        "models/multimodal_emotion_fusion.pth"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file)
            print(f"✅ {model_file} exists ({size} bytes)")
        else:
            print(f"❌ {model_file} missing")

def test_directories():
    """Test required directories"""
    print("\n=== Testing Directories ===")
    
    import os
    required_dirs = [
        "data",
        "models", 
        "vr_adaptation",
        "cascades",
        "logs"
    ]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ exists")
        else:
            print(f"❌ {dir_name}/ missing")

if __name__ == "__main__":
    print("VR Emotion Adaptation - Diagnostic Test")
    print("=" * 50)
    
    test_imports()
    test_custom_modules()
    test_hardware()
    test_ollama()
    test_model_files()
    test_directories()
    
    print("\n" + "=" * 50)
    print("Diagnostic test completed!")