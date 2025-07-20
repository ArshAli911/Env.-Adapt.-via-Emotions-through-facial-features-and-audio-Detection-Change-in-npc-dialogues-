import torch
import librosa
import numpy as np
import os

from models.emotion_models import AudioEmotionLSTM, EmotionClassifier
from data.data_loader import AudioEmotionDataset # To use its _map_emotion_code and emotion_mapping

# --- Configuration ---
SAMPLE_RATE = 22050
AUDIO_MODEL_WEIGHTS = "models/audio_emotion_lstm.pth"
AUDIO_DATA_PATH = "archive/audio_speech_actors_01-24" # Your audio dataset path

# --- Initialize Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
audio_classifier = EmotionClassifier('audio', str(device))

# Load dummy weights if they don't exist, otherwise load actual weights
# (This part assumes you have run main.py at least once to create dummy weights)
if os.path.exists(AUDIO_MODEL_WEIGHTS):
    audio_classifier.load_weights(AUDIO_MODEL_WEIGHTS)
else:
    print(f"Warning: No weights found at {AUDIO_MODEL_WEIGHTS}. "
          "Please run train.py or main.py first to create/train models.")

# --- Select an audio file for testing ---
# You can pick any .wav file from your archive/audio_speech_actors_01-24 directory.
# Example: Using one of the provided files
test_audio_file = os.path.join(AUDIO_DATA_PATH, "Actor_01", "03-01-01-01-01-01-01.wav")
# Make sure the file exists and the path is correct
if not os.path.exists(test_audio_file):
    print(f"Error: Test audio file not found at {test_audio_file}")
else:
    print(f"Testing with audio file: {test_audio_file}")
    
    # --- Load and preprocess audio ---
    audio, sr = librosa.load(test_audio_file, sr=SAMPLE_RATE)

    # Extract MFCC features (same logic as in data_loader.py and main.py)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Pad or truncate to a fixed length (e.g., 100 frames), as used in main.py
    target_length = 100
    if mfcc.shape[1] < target_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, target_length - mfcc.shape[1])), 'constant')
    else:
        mfcc = mfcc[:, :target_length]

    # Convert to tensor and permute for LSTM input (batch, sequence, feature)
    mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).permute(0, 2, 1)

    # --- Predict emotion ---
    prediction = audio_classifier.predict(mfcc_tensor.to(device))

    print(f"Predicted Emotion: {prediction['emotion']} (Confidence: {prediction['confidence']:.2f})")
    print(f"Probabilities: {prediction['probabilities']}")
