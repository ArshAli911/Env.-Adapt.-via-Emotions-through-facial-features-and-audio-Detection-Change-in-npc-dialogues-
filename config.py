"""
Configuration file for VR Emotion Adaptation System
Centralizes all system parameters and settings
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    """Configuration for emotion recognition models"""
    facial_model_weights: str = "models/facial_emotion_cnn.pth"
    audio_model_weights: str = "models/audio_emotion_lstm.pth"
    multimodal_model_weights: str = "models/multimodal_emotion_fusion.pth"
    emotion_labels: List[str] = None
    
    def __post_init__(self):
        if self.emotion_labels is None:
            self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@dataclass
class DataConfig:
    """Configuration for data paths and processing"""
    facial_data_path: str = "archive (1)/test"
    audio_data_path: str = "archive/audio_speech_actors_01-24"
    batch_size: int = 1
    sample_rate: int = 22050
    audio_chunk_size: int = 4096

@dataclass
class ProcessingConfig:
    """Configuration for real-time processing"""
    smoothing_window: int = 5
    confidence_threshold: float = 0.6
    frame_rate: int = 10  # Target FPS
    face_detection_scale: float = 1.1
    face_detection_min_neighbors: int = 4
    accuracy_update_interval: int = 30  # Update accuracy display every N frames
    enable_accuracy_tracking: bool = True
    enable_entropy_calculation: bool = True
    enable_stability_tracking: bool = True

@dataclass
class VRConfig:
    """Configuration for VR environment adaptation"""
    adaptation_speed: float = 0.1
    lighting_transition_time: float = 2.0
    audio_transition_time: float = 1.5
    npc_update_frequency: float = 0.5

@dataclass
class OllamaConfig:
    """Configuration for Ollama integration"""
    model_name: str = "mistral"
    max_retries: int = 3
    timeout: float = 10.0
    enable_text_analysis: bool = True
    enable_npc_dialogue: bool = True
    enable_contextual_info: bool = True

@dataclass
class LoggingConfig:
    """Configuration for logging and debugging"""
    log_level: str = "INFO"
    save_performance_graphs: bool = True
    save_emotion_history: bool = True
    output_directory: str = "output"

@dataclass
class SystemConfig:
    """Main system configuration"""
    model: ModelConfig = None
    data: DataConfig = None
    processing: ProcessingConfig = None
    vr: VRConfig = None
    ollama: OllamaConfig = None
    logging: LoggingConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()
        if self.vr is None:
            self.vr = VRConfig()
        if self.ollama is None:
            self.ollama = OllamaConfig()
        if self.logging is None:
            self.logging = LoggingConfig()

# Default configuration instance
config = SystemConfig()

def load_config_from_file(config_path: str) -> SystemConfig:
    """Load configuration from JSON file"""
    import json
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        # This would need more sophisticated parsing for nested dataclasses
        return config
    return config

def save_config_to_file(config: SystemConfig, config_path: str):
    """Save configuration to JSON file"""
    import json
    # Convert dataclass to dict (simplified)
    config_dict = {
        'model': config.model.__dict__,
        'data': config.data.__dict__,
        'processing': config.processing.__dict__,
        'vr': config.vr.__dict__,
        'ollama': config.ollama.__dict__,
        'logging': config.logging.__dict__
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2) 