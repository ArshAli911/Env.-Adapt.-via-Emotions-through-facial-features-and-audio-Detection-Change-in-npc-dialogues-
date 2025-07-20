"""
Advanced multimodal fusion techniques for emotion recognition
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
from enum import Enum

class FusionConfig:
    """Configuration for fusion models"""
    NUM_EMOTIONS = 7
    DEFAULT_HIDDEN_DIM = 128
    DEFAULT_DROPOUT = 0.3
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class AttentionFusion(nn.Module):
    """Attention-based fusion of facial and audio features"""
    
    def __init__(self, facial_dim: int, audio_dim: int, hidden_dim: int = FusionConfig.DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.facial_dim = facial_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        
        # Attention mechanisms
        self.facial_attention = nn.Sequential(
            nn.Linear(facial_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.audio_attention = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(facial_dim + audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(FusionConfig.DEFAULT_DROPOUT),
            nn.Linear(hidden_dim, FusionConfig.NUM_EMOTIONS)
        )
    
    def forward(self, facial_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        # Calculate attention weights
        facial_weight = self.facial_attention(facial_features)
        audio_weight = self.audio_attention(audio_features)
        
        # Apply attention
        weighted_facial = facial_features * facial_weight
        weighted_audio = audio_features * audio_weight
        
        # Concatenate and fuse
        combined = torch.cat([weighted_facial, weighted_audio], dim=1)
        return self.fusion(combined)

class TemporalEmotionTracker:
    """Track emotion changes over time with smoothing and trend analysis"""
    
    def __init__(self, window_size: int = 10, confidence_threshold: float = 0.6):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.emotion_history = []
        self.confidence_history = []
        self.probability_history = []
        self.trend_analysis = {}
        self.last_update_time = None
    
    def update(self, emotion: str, confidence: float, probabilities: np.ndarray):
        """Update emotion tracking with new prediction"""
        if not isinstance(emotion, str):
            raise TypeError("Emotion must be a string")
        if not isinstance(confidence, (int, float)):
            raise TypeError("Confidence must be a number")
        if not isinstance(probabilities, np.ndarray):
            raise TypeError("Probabilities must be a numpy array")
            
        # Validate emotion is in the list of valid emotions
        if emotion not in FusionConfig.EMOTION_LABELS and emotion != "unknown":
            raise ValueError(f"Invalid emotion: {emotion}. Must be one of {FusionConfig.EMOTION_LABELS} or 'unknown'")
            
        # Record current time for temporal analysis
        import time
        current_time = time.time()
        self.last_update_time = current_time
        
        # Store the new data
        self.emotion_history.append(emotion)
        self.confidence_history.append(confidence)
        self.probability_history.append(probabilities.copy())
        
        # Keep only recent history using deque-like behavior
        if len(self.emotion_history) > self.window_size:
            self.emotion_history.pop(0)
            self.confidence_history.pop(0)
            self.probability_history.pop(0)
        
        # Analyze trends
        self._analyze_trends()
    
    def get_stable_emotion(self) -> Tuple[str, float]:
        """Get the most stable emotion over the recent window"""
        if not self.emotion_history:
            return "neutral", 0.0
        
        # Use Counter for more efficient counting
        from collections import Counter
        emotion_counter = Counter(self.emotion_history)
        
        # Get most frequent emotion
        stable_emotion = emotion_counter.most_common(1)[0][0]
        stability_score = emotion_counter[stable_emotion] / len(self.emotion_history)
        
        return stable_emotion, stability_score
    
    def get_weighted_emotion(self) -> Tuple[str, float]:
        """Get emotion weighted by confidence scores"""
        if not self.emotion_history:
            return "neutral", 0.0
            
        # Weight emotions by their confidence
        emotion_weights = {}
        for emotion, confidence in zip(self.emotion_history, self.confidence_history):
            emotion_weights[emotion] = emotion_weights.get(emotion, 0) + confidence
            
        # Get emotion with highest weighted score
        weighted_emotion = max(emotion_weights, key=emotion_weights.get)
        total_weight = sum(emotion_weights.values())
        weighted_score = emotion_weights[weighted_emotion] / total_weight if total_weight > 0 else 0
        
        return weighted_emotion, weighted_score
    
    def get_smoothed_probabilities(self) -> np.ndarray:
        """Get temporally smoothed emotion probabilities"""
        if not self.probability_history:
            # Return uniform distribution if no history
            return np.ones(len(FusionConfig.EMOTION_LABELS)) / len(FusionConfig.EMOTION_LABELS)
            
        # Apply exponential weighting to recent probabilities
        weights = np.exp(np.linspace(-2, 0, len(self.probability_history)))
        weights = weights / weights.sum()  # Normalize weights
        
        # Apply weights to probability history
        smoothed = np.zeros_like(self.probability_history[0])
        for i, prob in enumerate(self.probability_history):
            smoothed += prob * weights[i]
            
        return smoothed
    
    def _analyze_trends(self):
        """Analyze emotion trends and patterns"""
        if len(self.emotion_history) < 3:
            return
        
        # Detect emotion transitions
        transitions = []
        for i in range(1, len(self.emotion_history)):
            if self.emotion_history[i] != self.emotion_history[i-1]:
                transitions.append((self.emotion_history[i-1], self.emotion_history[i]))
        
        # Calculate emotion volatility (how frequently emotions change)
        if len(self.emotion_history) > 1:
            changes = sum(1 for i in range(1, len(self.emotion_history)) 
                         if self.emotion_history[i] != self.emotion_history[i-1])
            volatility = changes / (len(self.emotion_history) - 1)
        else:
            volatility = 0.0
            
        # Calculate confidence trend
        if len(self.confidence_history) > 1:
            confidence_trend = self.confidence_history[-1] - self.confidence_history[0]
        else:
            confidence_trend = 0.0
        
        self.trend_analysis = {
            'transitions': transitions,
            'stability': 1.0 - (len(transitions) / len(self.emotion_history)),
            'dominant_emotion': max(set(self.emotion_history), key=self.emotion_history.count),
            'volatility': volatility,
            'confidence_trend': confidence_trend
        }