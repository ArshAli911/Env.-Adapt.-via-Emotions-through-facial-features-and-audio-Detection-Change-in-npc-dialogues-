"""
Unified configuration system for VR Emotion Adaptation
"""
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
from pathlib import Path


@dataclass
class ModelConfig:
    """Model-related configuration"""
    facial_model_weights: str = "models/facial_emotion_cnn.pth"
    audio_model_weights: str = "models/audio_emotion_lstm.pth"
    multimodal_model_weights: str = "models/multimodal_emotion_fusion.pth"
    
    face_input_size: Tuple[int, int] = (48, 48)
    mfcc_features: int = 13
    mfcc_target_length: int = 100
    
    emotion_labels: Tuple[str, ...] = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
    num_emotions: int = 7
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.num_emotions != len(self.emotion_labels):
            raise ValueError(f"num_emotions ({self.num_emotions}) must match emotion_labels length ({len(self.emotion_labels)})")
        
        if 'neutral' not in self.emotion_labels:
            raise ValueError("emotion_labels must contain 'neutral'")
        
        if self.mfcc_features <= 0 or self.mfcc_target_length <= 0:
            raise ValueError("MFCC parameters must be positive")
        
        if any(dim <= 0 for dim in self.face_input_size):
            raise ValueError("Face input size dimensions must be positive")
    
    @property
    def neutral_probs(self) -> np.ndarray:
        """Generate neutral emotion probabilities"""
        neutral_idx = self.emotion_labels.index('neutral')
        probs = np.full(self.num_emotions, 0.1)
        probs[neutral_idx] = 0.4
        return probs
    
    def validate_model_paths(self) -> bool:
        """Check if model weight files exist"""
        paths = [self.facial_model_weights, self.audio_model_weights, self.multimodal_model_weights]
        return all(Path(path).exists() for path in paths)


@dataclass
class ProcessingConfig:
    """Processing-related configuration"""
    sample_rate: int = 22050
    audio_chunk_size: int = 4096
    
    face_detection_scale: float = 1.1
    face_detection_min_neighbors: int = 4
    
    confidence_threshold: float = 0.6
    smoothing_window: int = 5
    processing_delay: float = 0.05
    
    dialogue_cooldown: float = 5.0
    accuracy_update_interval: int = 30
    
    def __post_init__(self):
        """Validate processing configuration"""
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.audio_chunk_size <= 0:
            raise ValueError("Audio chunk size must be positive")
        if not (1.0 <= self.face_detection_scale <= 2.0):
            raise ValueError("Face detection scale should be between 1.0 and 2.0")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        if self.smoothing_window <= 0:
            raise ValueError("Smoothing window must be positive")
        if self.processing_delay < 0:
            raise ValueError("Processing delay cannot be negative")


@dataclass
class UIConfig:
    """UI-related configuration"""
    autorefresh_interval_ms: int = 500
    queue_max_size: int = 50
    queue_process_limit: int = 5


@dataclass
class PathConfig:
    """Path-related configuration"""
    haarcascade_path: str = "cascades"
    logs_path: str = "logs"
    models_path: str = "models"


@dataclass
class UnifiedConfig:
    """Main configuration container"""
    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    ollama_model: str = "deepseek-r1:latest"


# Global configuration instance
config = UnifiedConfig()