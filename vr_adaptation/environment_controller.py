"""
VR Environment Adaptation Controller
Translates user emotions to environmental changes in VR worlds
"""

import numpy as np
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

from models.ollama_integration import OllamaIntegration

class EmotionType(Enum):
    """Enumeration of supported emotions"""
    ANGRY = "angry"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPY = "happy"
    NEUTRAL = "neutral"
    SAD = "sad"
    SURPRISE = "surprise"

@dataclass
class LightingConfig:
    """Configuration for lighting adaptation"""
    brightness: float  # 0.0 to 1.0
    temperature: float  # 2000K to 10000K (warm to cool)
    saturation: float  # 0.0 to 2.0
    contrast: float  # 0.5 to 2.0
    flicker_intensity: float  # 0.0 to 1.0
    shadow_intensity: float  # 0.0 to 1.0
    print(12345)

@dataclass
class AudioConfig:
    """Configuration for audio adaptation"""
    background_volume: float  # 0.0 to 1.0
    music_tempo: float  # 0.5 to 2.0 (slow to fast)
    music_key: str  # Major or minor
    ambient_sounds: List[str]  # List of ambient sound effects
    reverb_intensity: float  # 0.0 to 1.0
    bass_boost: float  # 0.0 to 1.0

@dataclass
class NPCConfig:
    """Configuration for NPC behavior adaptation"""
    friendliness: float  # -1.0 to 1.0 (hostile to friendly)
    approach_distance: float  # 1.0 to 10.0 meters
    dialogue_complexity: float  # 0.0 to 1.0
    response_speed: float  # 0.5 to 2.0
    interaction_frequency: float  # 0.0 to 1.0

class EnvironmentController:
    """Main controller for VR environment adaptation"""
    
    def __init__(self):
        self.current_emotion = EmotionType.NEUTRAL
        self.emotion_confidence = 0.0
        self.adaptation_speed = 0.1  # How quickly to transition between states
        self.ollama_client = OllamaIntegration(model="deepseek-r1")  # Use DeepSeek model
        
        # Predefined emotion configurations
        self.emotion_configs = self._initialize_emotion_configs()
        
        # Current environment state
        self.current_lighting = self.emotion_configs[EmotionType.NEUTRAL]['lighting']
        self.current_audio = self.emotion_configs[EmotionType.NEUTRAL]['audio']
        self.current_npc = self.emotion_configs[EmotionType.NEUTRAL]['npc']
    
    def _initialize_emotion_configs(self) -> Dict[EmotionType, Dict]:
        """Initialize predefined configurations for each emotion"""
        
        configs = {
            EmotionType.HAPPY: {
                'lighting': LightingConfig(
                    brightness=0.9,
                    temperature=4000,  # Warm
                    saturation=1.3,
                    contrast=1.2,
                    flicker_intensity=0.0,
                    shadow_intensity=0.2
                ),
                'audio': AudioConfig(
                    background_volume=0.7,
                    music_tempo=1.3,
                    music_key='major',
                    ambient_sounds=['birds', 'gentle_wind', 'laughter'],
                    reverb_intensity=0.3,
                    bass_boost=0.2
                ),
                'npc': NPCConfig(
                    friendliness=0.8,
                    approach_distance=2.0,
                    dialogue_complexity=0.8,
                    response_speed=1.2,
                    interaction_frequency=0.9
                )
            },
            
            EmotionType.SAD: {
                'lighting': LightingConfig(
                    brightness=0.4,
                    temperature=6000,  # Cool
                    saturation=0.7,
                    contrast=0.8,
                    flicker_intensity=0.0,
                    shadow_intensity=0.6
                ),
                'audio': AudioConfig(
                    background_volume=0.4,
                    music_tempo=0.7,
                    music_key='minor',
                    ambient_sounds=['rain', 'distant_thunder', 'wind'],
                    reverb_intensity=0.6,
                    bass_boost=0.1
                ),
                'npc': NPCConfig(
                    friendliness=0.6,
                    approach_distance=4.0,
                    dialogue_complexity=0.6,
                    response_speed=0.8,
                    interaction_frequency=0.4
                )
            },
            
            EmotionType.ANGRY: {
                'lighting': LightingConfig(
                    brightness=0.8,
                    temperature=3000,  # Very warm/red
                    saturation=1.5,
                    contrast=1.8,
                    flicker_intensity=0.3,
                    shadow_intensity=0.4
                ),
                'audio': AudioConfig(
                    background_volume=0.8,
                    music_tempo=1.8,
                    music_key='minor',
                    ambient_sounds=['drum_beats', 'electric_guitar', 'crowd'],
                    reverb_intensity=0.2,
                    bass_boost=0.8
                ),
                'npc': NPCConfig(
                    friendliness=-0.3,
                    approach_distance=6.0,
                    dialogue_complexity=0.3,
                    response_speed=1.5,
                    interaction_frequency=0.2
                )
            },
            
            EmotionType.FEAR: {
                'lighting': LightingConfig(
                    brightness=0.3,
                    temperature=7000,  # Very cool
                    saturation=0.5,
                    contrast=1.5,
                    flicker_intensity=0.7,
                    shadow_intensity=0.9
                ),
                'audio': AudioConfig(
                    background_volume=0.6,
                    music_tempo=0.6,
                    music_key='minor',
                    ambient_sounds=['distant_echoes', 'creaking_wood', 'whispers'],
                    reverb_intensity=0.8,
                    bass_boost=0.4
                ),
                'npc': NPCConfig(
                    friendliness=0.2,
                    approach_distance=8.0,
                    dialogue_complexity=0.4,
                    response_speed=0.6,
                    interaction_frequency=0.1
                )
            },
            
            EmotionType.SURPRISE: {
                'lighting': LightingConfig(
                    brightness=1.0,
                    temperature=5000,  # Neutral
                    saturation=1.2,
                    contrast=1.4,
                    flicker_intensity=0.5,
                    shadow_intensity=0.3
                ),
                'audio': AudioConfig(
                    background_volume=0.9,
                    music_tempo=1.5,
                    music_key='major',
                    ambient_sounds=['sudden_impact', 'glass_breaking', 'alarm'],
                    reverb_intensity=0.4,
                    bass_boost=0.6
                ),
                'npc': NPCConfig(
                    friendliness=0.4,
                    approach_distance=3.0,
                    dialogue_complexity=0.7,
                    response_speed=1.8,
                    interaction_frequency=0.6
                )
            },
            
            EmotionType.DISGUST: {
                'lighting': LightingConfig(
                    brightness=0.6,
                    temperature=4500,  # Slightly warm
                    saturation=0.8,
                    contrast=1.1,
                    flicker_intensity=0.2,
                    shadow_intensity=0.5
                ),
                'audio': AudioConfig(
                    background_volume=0.5,
                    music_tempo=0.8,
                    music_key='minor',
                    ambient_sounds=['dripping_water', 'rustling_leaves', 'distant_cries'],
                    reverb_intensity=0.5,
                    bass_boost=0.3
                ),
                'npc': NPCConfig(
                    friendliness=-0.1,
                    approach_distance=5.0,
                    dialogue_complexity=0.5,
                    response_speed=0.9,
                    interaction_frequency=0.3
                )
            },
            
            EmotionType.NEUTRAL: {
                'lighting': LightingConfig(
                    brightness=0.7,
                    temperature=5000,  # Neutral
                    saturation=1.0,
                    contrast=1.0,
                    flicker_intensity=0.0,
                    shadow_intensity=0.4
                ),
                'audio': AudioConfig(
                    background_volume=0.6,
                    music_tempo=1.0,
                    music_key='major',
                    ambient_sounds=['gentle_wind', 'birds', 'water_stream'],
                    reverb_intensity=0.3,
                    bass_boost=0.2
                ),
                'npc': NPCConfig(
                    friendliness=0.5,
                    approach_distance=3.5,
                    dialogue_complexity=0.7,
                    response_speed=1.0,
                    interaction_frequency=0.6
                )
            }
        }
        
        return configs
    
    def update_emotion(self, emotion: str, confidence: float):
        """Update the current emotion and confidence"""
        try:
            self.current_emotion = EmotionType(emotion.lower())
            self.emotion_confidence = confidence
        except ValueError:
            print(f"Unknown emotion: {emotion}")
            return
    
    def get_environment_config(self) -> Dict:
        """Get current environment configuration"""
        target_config = self.emotion_configs[self.current_emotion]
        
        # Apply confidence-based blending
        if self.emotion_confidence < 0.5:
            # Blend with neutral configuration
            neutral_config = self.emotion_configs[EmotionType.NEUTRAL]
            blend_factor = self.emotion_confidence * 2  # 0 to 1
            
            return {
                'lighting': self._blend_lighting(target_config['lighting'], neutral_config['lighting'], blend_factor),
                'audio': self._blend_audio(target_config['audio'], neutral_config['audio'], blend_factor),
                'npc': self._blend_npc(target_config['npc'], neutral_config['npc'], blend_factor)
            }
        
        return {
            'lighting': target_config['lighting'],
            'audio': target_config['audio'],
            'npc': target_config['npc']
        }
    
    def _blend_lighting(self, config1: LightingConfig, config2: LightingConfig, factor: float) -> LightingConfig:
        """Blend two lighting configurations"""
        return LightingConfig(
            brightness=config1.brightness * factor + config2.brightness * (1 - factor),
            temperature=config1.temperature * factor + config2.temperature * (1 - factor),
            saturation=config1.saturation * factor + config2.saturation * (1 - factor),
            contrast=config1.contrast * factor + config2.contrast * (1 - factor),
            flicker_intensity=config1.flicker_intensity * factor + config2.flicker_intensity * (1 - factor),
            shadow_intensity=config1.shadow_intensity * factor + config2.shadow_intensity * (1 - factor)
        )
    
    def _blend_audio(self, config1: AudioConfig, config2: AudioConfig, factor: float) -> AudioConfig:
        """Blend two audio configurations"""
        return AudioConfig(
            background_volume=config1.background_volume * factor + config2.background_volume * (1 - factor),
            music_tempo=config1.music_tempo * factor + config2.music_tempo * (1 - factor),
            music_key=config1.music_key if factor > 0.5 else config2.music_key,
            ambient_sounds=config1.ambient_sounds if factor > 0.5 else config2.ambient_sounds,
            reverb_intensity=config1.reverb_intensity * factor + config2.reverb_intensity * (1 - factor),
            bass_boost=config1.bass_boost * factor + config2.bass_boost * (1 - factor)
        )
    
    def _blend_npc(self, config1: NPCConfig, config2: NPCConfig, factor: float) -> NPCConfig:
        """Blend two NPC configurations"""
        return NPCConfig(
            friendliness=config1.friendliness * factor + config2.friendliness * (1 - factor),
            approach_distance=config1.approach_distance * factor + config2.approach_distance * (1 - factor),
            dialogue_complexity=config1.dialogue_complexity * factor + config2.dialogue_complexity * (1 - factor),
            response_speed=config1.response_speed * factor + config2.response_speed * (1 - factor),
            interaction_frequency=config1.interaction_frequency * factor + config2.interaction_frequency * (1 - factor)
        )
    
    def get_unity_parameters(self) -> Dict:
        """Returns the current environment configuration as a dictionary for Unity"""
        unity_params = {
            "lighting": {
                "brightness": self.current_lighting.brightness,
                "temperature": self.current_lighting.temperature,
                "saturation": self.current_lighting.saturation,
                "contrast": self.current_lighting.contrast,
                "flickerIntensity": self.current_lighting.flicker_intensity,
                "shadowIntensity": self.current_lighting.shadow_intensity
            },
            "audio": {
                "backgroundVolume": self.current_audio.background_volume,
                "musicTempo": self.current_audio.music_tempo,
                "musicKey": self.current_audio.music_key,
                "ambientSounds": self.current_audio.ambient_sounds,
                "reverbIntensity": self.current_audio.reverb_intensity,
                "bassBoost": self.current_audio.bass_boost
            },
            "npc": {
                "friendliness": self.current_npc.friendliness,
                "approachDistance": self.current_npc.approach_distance,
                "dialogueComplexity": self.current_npc.dialogue_complexity,
                "responseSpeed": self.current_npc.response_speed,
                "interactionFrequency": self.current_npc.interaction_frequency
            },
            "currentEmotion": self.current_emotion.value,
            "emotionConfidence": self.emotion_confidence
        }
        return unity_params

    def analyze_text_emotion_with_ollama(self, text: str) -> str:
        """Analyzes text to deduce an emotion using Ollama."""
        return self.ollama_client.analyze_text_emotion(text)

    def generate_dialogue(self, emotion: str) -> str:
        """Generate dialogue based on emotion using Ollama."""
        context = f"The user is currently expressing {emotion} emotion. Respond with a supportive or appropriate short dialogue."
        if not self.ollama_client.is_model_available():
            return "Ollama model is not available for dialogue generation."

        try:
            dialogue = self.ollama_client.generate_npc_dialogue(emotion, context)
            if dialogue is None:
                return "I'm not sure how to respond right now."
            return dialogue
        except Exception as e:
            print(f"Ollama dialogue generation failed: {e}")
            print(11111)
            return f"I'm here to support you. (Error: {str(e)[:50]}...)"

    def generate_contextual_info_with_ollama(self, emotion: str) -> str:
        """Generates descriptive contextual information for the VR environment using Ollama."""
        environment_state = self.get_environment_config()
        return self.ollama_client.generate_contextual_info(environment_state, emotion)

    def save_config(self, filename: str):
        """Save current configuration to file"""
        config = self.get_unity_parameters()
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, filename: str):
        """Load configuration from file"""
        with open(filename, 'r') as f:
            config = json.load(f)
        
        # Update emotion state
        emotion_name = config['currentEmotion']
        self.current_emotion = EmotionType(emotion_name)
        self.emotion_confidence = config['emotionConfidence']

if __name__ == "__main__":
    # Example Usage:
    controller = EnvironmentController()

    # Simulate emotion update
    controller.update_emotion("happy", 0.85)
    print("\n--- Environment state after happy emotion ---")
    config = controller.get_unity_parameters()
    print(f"Lighting - Brightness: {config['lighting']['brightness']:.2f}, Temperature: {config['lighting']['temperature']}K")
    print(f"Audio - Volume: {config['audio']['backgroundVolume']:.2f}, Tempo: {config['audio']['musicTempo']:.2f}")
    print(f"NPC - Friendliness: {config['npc']['friendliness']:.2f}, Distance: {config['npc']['approachDistance']:.1f}m")

    # Test confidence blending
    controller.update_emotion("sad", 0.6)
    print("\n--- Environment state after sad emotion (blended) ---")
    config = controller.get_unity_parameters()
    print(f"Lighting - Brightness: {config['lighting']['brightness']:.2f}, Temperature: {config['lighting']['temperature']}K")
    print(f"Audio - Volume: {config['audio']['backgroundVolume']:.2f}, Tempo: {config['audio']['musicTempo']:.2f}")
    print(f"NPC - Friendliness: {config['npc']['friendliness']:.2f}, Distance: {config['npc']['approachDistance']:.1f}m")

    # Test saving and loading config
    test_filename = "test_env_config.json"
    controller.save_config(test_filename)
    print(f"\nSaved current config to {test_filename}")

    new_controller = EnvironmentController()
    new_controller.load_config(test_filename)
    print(f"Loaded config from {test_filename}")
    loaded_config = new_controller.get_unity_parameters()
    print(f"Loaded Emotion: {loaded_config['currentEmotion']}, Confidence: {loaded_config['emotionConfidence']:.2f}")
    print(f"Loaded Lighting Brightness: {loaded_config['lighting']['brightness']:.2f}")
    print(f"Loaded Audio Volume: {loaded_config['audio']['backgroundVolume']:.2f}")
    print(f"Loaded NPC Friendliness: {loaded_config['npc']['friendliness']:.2f}")

    # Test Ollama integrations
    print("\n--- Testing Ollama Integrations ---")
    ollama_test_text = "I am feeling very anxious about this presentation."
    ollama_emotion = controller.analyze_text_emotion_with_ollama(ollama_test_text)
    print(f"Ollama detected emotion from text \"{ollama_test_text}\": {ollama_emotion}")

    ollama_npc_dialogue = controller.generate_dialogue(ollama_emotion)
    print(f"Ollama generated NPC dialogue for {ollama_emotion} emotion: {ollama_npc_dialogue}")

    ollama_contextual_info = controller.generate_contextual_info_with_ollama(ollama_emotion)
    print(f"Ollama generated contextual info for {ollama_emotion} emotion: {ollama_contextual_info}") 