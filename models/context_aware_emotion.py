"""
Context-aware emotion recognition system
"""
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ContextConfig:
    """Configuration constants for context-aware emotion recognition"""
    BASE_EMOTION_DIM = 7
    # Calculate context dimension dynamically based on enum sizes
    TIME_PERIOD_DIM = 4  # TimePeriod enum size
    ACTIVITY_TYPE_DIM = 4  # ActivityType enum size  
    ENVIRONMENT_TYPE_DIM = 5  # EnvironmentType enum size
    SESSION_DURATION_DIM = 1  # Single normalized value
    EMOTION_HISTORY_DIM = 7  # EmotionLabel enum size
    CONTEXT_DIM = TIME_PERIOD_DIM + ACTIVITY_TYPE_DIM + ENVIRONMENT_TYPE_DIM + SESSION_DURATION_DIM + EMOTION_HISTORY_DIM
    
    CONTEXT_HIDDEN_DIM = 64
    CONTEXT_INTERMEDIATE_DIM = 32
    ADJUSTMENT_HIDDEN_DIM = 64
    MAX_PATTERNS_PER_EMOTION = 100
    PROBABILITY_TOLERANCE = 1.1  # Allow small floating point errors
    SESSION_DURATION_NORMALIZATION = 60.0  # Normalize to hours

class TimePeriod(Enum):
    """Time periods for contextual emotion recognition"""
    MORNING = "morning"
    AFTERNOON = "afternoon" 
    EVENING = "evening"
    NIGHT = "night"

class ActivityType(Enum):
    """Activity types for contextual emotion recognition"""
    GAMING = "gaming"
    WORKING = "working"
    RELAXING = "relaxing"
    UNKNOWN = "unknown"

class EnvironmentType(Enum):
    """Environment types for contextual emotion recognition"""
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    QUIET = "quiet"
    NOISY = "noisy"
    UNKNOWN = "unknown"

class EmotionLabel(Enum):
    """Standard emotion labels"""
    ANGRY = "angry"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPY = "happy"
    NEUTRAL = "neutral"
    SAD = "sad"
    SURPRISE = "surprise"

@dataclass
class EmotionContext:
    """Context information for emotion recognition"""
    time_of_day: TimePeriod
    activity_type: ActivityType
    environment: EnvironmentType
    user_history: Dict[EmotionLabel, float]  # recent emotion patterns
    session_duration: float  # minutes in current session
    
    def __post_init__(self):
        """Validate context data after initialization"""
        if self.session_duration < 0:
            raise ValueError("Session duration cannot be negative")
        if self.user_history and not (0 <= sum(self.user_history.values()) <= ContextConfig.PROBABILITY_TOLERANCE):
            raise ValueError("User history probabilities should sum to approximately 1.0")

class ContextualEmotionClassifier(nn.Module):
    """Emotion classifier that considers contextual information"""
    
    def __init__(self, 
                 base_emotion_dim: int = ContextConfig.BASE_EMOTION_DIM, 
                 context_dim: int = ContextConfig.CONTEXT_DIM,
                 dropout_rate: float = 0.3):
        super().__init__()
        self.base_emotion_dim = base_emotion_dim
        self.context_dim = context_dim
        
        # Context encoding with dropout for regularization
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, ContextConfig.CONTEXT_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ContextConfig.CONTEXT_HIDDEN_DIM, ContextConfig.CONTEXT_INTERMEDIATE_DIM),
            nn.ReLU()
        )
        
        # Contextual adjustment layer
        self.context_adjustment = nn.Sequential(
            nn.Linear(ContextConfig.CONTEXT_INTERMEDIATE_DIM + base_emotion_dim, ContextConfig.ADJUSTMENT_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ContextConfig.ADJUSTMENT_HIDDEN_DIM, base_emotion_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, base_emotions: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        """Apply contextual adjustment to base emotion predictions"""
        context_encoded = self.context_encoder(context_features)
        combined = torch.cat([base_emotions, context_encoded], dim=1)
        adjusted_emotions = self.context_adjustment(combined)
        return adjusted_emotions

class ContextManager:
    """Manages contextual information for emotion recognition"""
    
    def __init__(self):
        self.current_context = None
        self.context_history = []
        self.user_patterns = {}
    
    def update_context(self, activity: Optional[ActivityType] = None, environment: Optional[EnvironmentType] = None) -> EmotionContext:
        """Update current context information"""
        current_time = datetime.now().time()
        time_of_day = self._get_time_period(current_time)
        
        # Calculate session duration (placeholder)
        session_duration = 0.0  # Would be calculated from session start
        
        self.current_context = EmotionContext(
            time_of_day=time_of_day,
            activity_type=activity or ActivityType.UNKNOWN,
            environment=environment or EnvironmentType.UNKNOWN,
            user_history=self._convert_user_patterns_to_enum_keys(),
            session_duration=session_duration
        )
        
        return self.current_context
    
    def _convert_user_patterns_to_enum_keys(self) -> Dict[EmotionLabel, float]:
        """Convert string-based user patterns to enum-based dictionary"""
        enum_patterns = {}
        for emotion_str, pattern_list in self.user_patterns.items():
            try:
                emotion_enum = EmotionLabel(emotion_str)
                # Calculate average confidence for this emotion
                if pattern_list:
                    avg_confidence = sum(p['confidence'] for p in pattern_list) / len(pattern_list)
                    enum_patterns[emotion_enum] = avg_confidence
            except ValueError:
                # Skip invalid emotion strings
                continue
        return enum_patterns
    
    def _get_time_period(self, current_time: time) -> TimePeriod:
        """Determine time period from current time"""
        hour = current_time.hour
        if 5 <= hour < 12:
            return TimePeriod.MORNING
        elif 12 <= hour < 17:
            return TimePeriod.AFTERNOON
        elif 17 <= hour < 21:
            return TimePeriod.EVENING
        else:
            return TimePeriod.NIGHT
    
    def encode_context(self, context: EmotionContext) -> np.ndarray:
        """Encode context into numerical features using structured approach"""
        features = np.zeros(ContextConfig.CONTEXT_DIM)
        current_idx = 0
        
        # Time of day encoding (one-hot)
        time_features = self._encode_enum_one_hot(context.time_of_day, TimePeriod)
        features[current_idx:current_idx + len(time_features)] = time_features
        current_idx += len(time_features)
        
        # Activity type encoding (one-hot) 
        activity_features = self._encode_enum_one_hot(context.activity_type, ActivityType)
        features[current_idx:current_idx + len(activity_features)] = activity_features
        current_idx += len(activity_features)
        
        # Environment type encoding (one-hot)
        env_features = self._encode_enum_one_hot(context.environment, EnvironmentType)
        features[current_idx:current_idx + len(env_features)] = env_features
        current_idx += len(env_features)
        
        # Session duration (normalized)
        features[current_idx] = min(
            context.session_duration / ContextConfig.SESSION_DURATION_NORMALIZATION, 1.0
        )
        current_idx += 1
        
        # User history (emotion frequencies)
        for emotion_label in EmotionLabel:
            features[current_idx] = context.user_history.get(emotion_label, 0.0)
            current_idx += 1
        
        return features
    
    def _encode_enum_one_hot(self, value: Enum, enum_class: type) -> np.ndarray:
        """Helper method to create one-hot encoding for enum values"""
        enum_values = list(enum_class)
        one_hot = np.zeros(len(enum_values))
        try:
            index = enum_values.index(value)
            one_hot[index] = 1.0
        except ValueError:
            # Handle unknown values gracefully
            pass
        return one_hot
    
    def learn_user_patterns(self, emotion: str, confidence: float):
        """Learn user emotion patterns over time"""
        # Validate inputs
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
        
        # Validate emotion is a known emotion label
        try:
            emotion_enum = EmotionLabel(emotion)
            emotion_key = emotion_enum.value
        except ValueError:
            raise ValueError(f"Unknown emotion '{emotion}'. Valid emotions: {[e.value for e in EmotionLabel]}")
        
        if emotion_key not in self.user_patterns:
            self.user_patterns[emotion_key] = []
        
        self.user_patterns[emotion_key].append({
            'confidence': confidence,
            'timestamp': datetime.now(),
            'context': self.current_context
        })
        
        # Keep only recent patterns using configuration constant
        if len(self.user_patterns[emotion_key]) > ContextConfig.MAX_PATTERNS_PER_EMOTION:
            self.user_patterns[emotion_key] = self.user_patterns[emotion_key][-ContextConfig.MAX_PATTERNS_PER_EMOTION:]