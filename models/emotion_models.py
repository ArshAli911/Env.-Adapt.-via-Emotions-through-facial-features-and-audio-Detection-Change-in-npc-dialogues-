"""
Neural network models for emotion recognition
Includes models for facial expressions and audio analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class FacialEmotionCNN(nn.Module):
    """CNN model for facial emotion recognition"""
    
    def __init__(self, num_classes=7, input_channels=1):
        super(FacialEmotionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Convolutional layers with batch norm and ReLU
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class AudioEmotionLSTM(nn.Module):
    """LSTM model for audio emotion recognition"""
    
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, num_classes=7, dropout=0.5):
        super(AudioEmotionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(attended_output))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class MultiModalEmotionFusion(nn.Module):
    """Multi-modal fusion model combining facial and audio features"""
    
    def __init__(self, facial_features=256, audio_features=128, num_classes=7, fusion_dim=512):
        super(MultiModalEmotionFusion, self).__init__()
        
        # Feature extraction networks
        self.facial_extractor = FacialEmotionCNN(num_classes=facial_features)
        self.audio_extractor = AudioEmotionLSTM(num_classes=audio_features)
        
        # Fusion layers
        self.fusion_fc1 = nn.Linear(facial_features + audio_features, fusion_dim)
        self.fusion_fc2 = nn.Linear(fusion_dim, fusion_dim // 2)
        self.fusion_fc3 = nn.Linear(fusion_dim // 2, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, facial_input, audio_input):
        # Extract features from each modality
        print(f"[DEBUG] MultiModalEmotionFusion - facial_input type: {type(facial_input)}")
        print(f"[DEBUG] MultiModalEmotionFusion - facial_input shape: {facial_input.shape}")
        facial_features = self.facial_extractor(facial_input)
        audio_features = self.audio_extractor(audio_input)
        
        # Concatenate features
        combined_features = torch.cat([facial_features, audio_features], dim=1)
        
        # Fusion layers
        x = F.relu(self.fusion_fc1(combined_features))
        x = self.dropout(x)
        x = F.relu(self.fusion_fc2(x))
        x = self.dropout(x)
        x = self.fusion_fc3(x)
        
        return x

class EmotionClassifier:
    """Wrapper class for emotion classification"""
    
    def __init__(self, model_type='facial', device='cpu', model=None):
        self.model_type = model_type
        self.device = device
        
        if model is not None:
            self.model = model.to(device)
        else:
            if model_type == 'facial':
                self.model = FacialEmotionCNN().to(device)
            elif model_type == 'audio':
                self.model = AudioEmotionLSTM().to(device)
            elif model_type == 'multimodal':
                self.model = MultiModalEmotionFusion().to(device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
    def predict(self, input_data):
        """Predict emotion from input data"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(input_data, (list, tuple)):
                # Multi-modal input
                # Ensure all inputs are tensors and on the correct device
                processed_input = []
                for item in input_data:
                    if isinstance(item, (int, float, bool)): # Handle scalar values
                        processed_input.append(torch.tensor(item, dtype=torch.float32, device=self.device))
                    elif isinstance(item, np.ndarray):
                        processed_input.append(torch.from_numpy(item).float().to(self.device))
                    else:
                        processed_input.append(item.to(self.device) if hasattr(item, 'to') else item)
                outputs = self.model(*processed_input)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            else:
                # Single modal input
                if isinstance(input_data, torch.Tensor):
                    input_data_tensor = input_data.detach().clone().to(self.device)
                else:
                    input_data_tensor = torch.tensor(input_data, dtype=torch.float32, device=self.device)
                outputs = self.model(input_data_tensor) # Use the potentially converted tensor
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy() # Keep as 2D array for easier handling if batch

            # Handle batch predictions
            if probabilities.shape[0] == 1: # Check if batch size is 1
                # Single prediction
                predicted_class_idx = np.argmax(probabilities[0])
                emotion_name = self.emotion_labels[predicted_class_idx]
                confidence = np.max(probabilities[0])
            else:
                # Batch prediction - take the first one for compatibility
                predicted_class_idx = np.argmax(probabilities[0])
                emotion_name = self.emotion_labels[predicted_class_idx]
                confidence = np.max(probabilities[0])

            return {
                'emotion': emotion_name,
                'confidence': confidence,
                'probabilities': probabilities.flatten() # Flatten only for the return value
            }
    
    def load_weights(self, weights_path):
        """Load pre-trained weights"""
        if os.path.exists(weights_path):
            # Map weights to the correct device (CPU or CUDA)
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded weights from {weights_path} to {self.device}")
        else:
            raise FileNotFoundError(f"Weights file not found at {weights_path}")
    
    def save_weights(self, weights_path):
        """Save model weights"""
        torch.save(self.model.state_dict(), weights_path)
        print(f"Saved weights to {weights_path}")

def create_model(model_type, device='cpu'):
    """Factory function to create emotion recognition models"""
    
    if model_type == 'facial':
        return FacialEmotionCNN().to(device)
    elif model_type == 'audio':
        return AudioEmotionLSTM().to(device)
    elif model_type == 'multimodal':
        return MultiModalEmotionFusion().to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test facial model
    facial_model = FacialEmotionCNN()
    test_input = torch.randn(1, 1, 48, 48)
    facial_output = facial_model(test_input)
    print(f"Facial model output shape: {facial_output.shape}")
    
    # Test audio model
    audio_model = AudioEmotionLSTM()
    test_audio = torch.randn(1, 100, 13)  # batch_size, sequence_length, features
    audio_output = audio_model(test_audio)  
    print(f"Audio model output shape: {audio_output.shape}")
    
    # Test multimodal model
    multimodal_model = MultiModalEmotionFusion()
    multimodal_output = multimodal_model(test_input, test_audio)
    print(f"Multimodal model output shape: {multimodal_output.shape}") 