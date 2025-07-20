"""
Data loader for emotion recognition datasets
Handles both facial expressions and audio data
"""

import os
import numpy as np
import pandas as pd
import cv2
import librosa
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

class EmotionDataset:
    """Base class for emotion datasets"""
    
    def __init__(self, data_path, emotion_mapping=None):
        self.data_path = data_path
        self.emotion_mapping = emotion_mapping or {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
            'neutral': 4, 'sad': 5, 'surprise': 6
        }
        self.reverse_mapping = {v: k for k, v in self.emotion_mapping.items()}
        
    def get_emotion_label(self, emotion_name):
        """Convert emotion name to numerical label"""
        return self.emotion_mapping.get(emotion_name.lower(), -1)
    
    def get_emotion_name(self, label):
        """Convert numerical label to emotion name"""
        return self.reverse_mapping.get(label, 'unknown')

class FacialExpressionDataset(Dataset):
    """Dataset for facial expression recognition"""
    
    def __init__(self, data_path, transform=None, mode='train'):
        self.data_path = data_path
        self.transform = transform
        self.mode = mode
        self.images = []
        self.labels = []
        
        # Load data from directory structure
        self._load_data()
        
    def _load_data(self):
        """Load images and labels from directory structure"""
        emotion_folders = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        for emotion_idx, emotion in enumerate(emotion_folders):
            emotion_path = os.path.join(self.data_path, emotion)
            if not os.path.exists(emotion_path):
                continue
                
            for filename in os.listdir(emotion_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(emotion_path, filename)
                    self.images.append(image_path)
                    self.labels.append(emotion_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        image_path = self.images[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            # Handle corrupted images
            image = np.zeros((48, 48), dtype=np.uint8)
        
        # Resize to standard size
        image = cv2.resize(image, (48, 48))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor
        image = torch.FloatTensor(image).unsqueeze(0)  # Add channel dimension
        label = torch.LongTensor([self.labels[idx]])
        
        return image, label

class AudioEmotionDataset(Dataset):
    """Dataset for audio emotion recognition"""
    
    def __init__(self, data_path, sample_rate=22050, max_length=5.0, transform=None):
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.transform = transform
        self.audio_files = []
        self.labels = []
        
        # Load data from directory structure
        self._load_data()
        
    def _load_data(self):
        """Load audio files and labels from directory structure"""
        actor_folders = [f for f in os.listdir(self.data_path) if f.startswith('Actor_')]
        
        for actor_folder in actor_folders:
            actor_path = os.path.join(self.data_path, actor_folder)
            if not os.path.isdir(actor_path):
                continue
                
            for filename in os.listdir(actor_path):
                if filename.lower().endswith('.wav'):
                    # Extract emotion from filename (assuming format: XX-XX-XX-XX-XX-XX-XX.wav)
                    parts = filename.split('-')
                    if len(parts) >= 7:
                        emotion_code = parts[6].split('.')[0]  # Remove .wav extension
                        # Map emotion code to label (this mapping needs to be defined based on your dataset)
                        emotion_label = self._map_emotion_code(emotion_code)
                        
                        if emotion_label is not None:
                            audio_path = os.path.join(actor_path, filename)
                            self.audio_files.append(audio_path)
                            self.labels.append(emotion_label)
    
    def _map_emotion_code(self, emotion_code):
        """Map emotion code from filename to numerical label"""
        # This mapping should be adjusted based on your specific dataset format
        emotion_mapping = {
            '01': 0,  # angry
            '02': 1,  # disgust
            '03': 2,  # fear
            '04': 3,  # happy
            '05': 4,  # neutral
            '06': 5,  # sad
            '07': 6   # surprise
        }
        return emotion_mapping.get(emotion_code, None)
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load audio file
        audio_path = self.audio_files[idx]
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            # Return dummy data for corrupted files to avoid crashing
            audio = np.zeros(int(self.sample_rate * self.max_length), dtype=np.float32)
            sr = self.sample_rate
        
        # Pad or truncate to max_length
        max_samples = int(self.sample_rate * self.max_length)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        else:
            audio = np.pad(audio, (0, max_samples - len(audio)), 'constant')
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # Normalize
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        
        # Transpose so shape is [time, 13] for LSTM
        mfcc = torch.FloatTensor(mfcc.T)
        label = torch.LongTensor([self.labels[idx]])
        
        return mfcc, label

def create_data_loaders(facial_data_path, audio_data_path, batch_size=32, test_size=0.2):
    """Create data loaders for both facial and audio datasets"""
    
    # Create facial expression datasets
    facial_dataset = FacialExpressionDataset(facial_data_path, mode='train')
    facial_train, facial_test = train_test_split(
        range(len(facial_dataset)), test_size=test_size, random_state=42, stratify=facial_dataset.labels
    )
    
    # Create audio datasets
    audio_dataset = AudioEmotionDataset(audio_data_path)
    audio_train, audio_test = train_test_split(
        range(len(audio_dataset)), test_size=test_size, random_state=42, stratify=audio_dataset.labels
    )
    
    # Create data loaders
    facial_train_loader = DataLoader(
        [facial_dataset[i] for i in facial_train],
        batch_size=batch_size,
        shuffle=True
    )
    
    facial_test_loader = DataLoader(
        [facial_dataset[i] for i in facial_test],
        batch_size=batch_size,
        shuffle=False
    )
    
    audio_train_loader = DataLoader(
        [audio_dataset[i] for i in audio_train],
        batch_size=batch_size,
        shuffle=True
    )
    
    audio_test_loader = DataLoader(
        [audio_dataset[i] for i in audio_test],
        batch_size=batch_size,
        shuffle=False
    )
    
    return {
        'facial_train': facial_train_loader,
        'facial_test': facial_test_loader,
        'audio_train': audio_train_loader,
        'audio_test': audio_test_loader
    }

if __name__ == "__main__":
    # Test data loading
    facial_path = "archive (1)/train"
    audio_path = "archive/audio_speech_actors_01-24"
    
    try:
        loaders = create_data_loaders(facial_path, audio_path, batch_size=16)
        print("Data loaders created successfully!")
        print(f"Facial train batches: {len(loaders['facial_train'])}")
        print(f"Audio train batches: {len(loaders['audio_train'])}")
    except Exception as e:
        print(f"Error creating data loaders: {e}") 