# API Documentation

## VR Emotion Adaptation System API Reference

This document provides comprehensive API documentation for the VR Emotion Adaptation System components.

## Core Classes

### EmotionRecognitionSystem

Main system class that orchestrates emotion detection and VR adaptation.

```python
from vr_emotion_adaptation import EmotionRecognitionSystem

system = EmotionRecognitionSystem()
```

#### Methods

##### `start_processing()`
Starts the real-time emotion detection and VR adaptation loop.

```python
system.start_processing()
```

##### `stop_processing()`
Stops the processing loop and releases resources.

```python
system.stop_processing()
```

##### `get_current_emotion()`
Returns the current detected emotion with confidence score.

```python
emotion_data = system.get_current_emotion()
# Returns: {'emotion': 'happy', 'confidence': 0.85, 'probabilities': [...]}
```

### EnvironmentController

Handles VR environment parameter adaptation based on detected emotions.

```python
from vr_adaptation.environment_controller import EnvironmentController

controller = EnvironmentController()
```

#### Methods

##### `adapt_environment(emotion, confidence)`
Adapts VR environment parameters based on emotion.

```python
vr_params = controller.adapt_environment('happy', 0.8)
# Returns: VR parameter dictionary
```

##### `get_unity_parameters()`
Returns Unity-compatible parameter format.

```python
unity_params = controller.get_unity_parameters()
```

### OllamaIntegration

AI-powered NPC dialogue generation using Ollama.

```python
from models.ollama_integration import OllamaIntegration

ollama = OllamaIntegration(model="deepseek-r1:latest")
```

#### Methods

##### `generate_npc_dialogue(emotion, context="")`
Generates contextual NPC dialogue based on emotion.

```python
dialogue = ollama.generate_npc_dialogue('sad', 'User looks upset')
# Returns: "I'm here for you. Everything will be okay."
```

##### `is_model_available()`
Checks if the Ollama model is available.

```python
available = ollama.is_model_available()
# Returns: True/False
```

## Configuration Classes

### EmotionConfig

Core system configuration parameters.

```python
from config import EmotionConfig

config = EmotionConfig()
```

#### Properties

- `EMOTION_LABELS`: List of supported emotions
- `MODEL_PATHS`: Paths to trained models
- `PROCESSING_PARAMS`: Processing parameters
- `DEVICE_CONFIG`: GPU/CPU configuration

### VRAdaptationConfig

VR environment adaptation parameters.

```python
from config_unified import VRAdaptationConfig

vr_config = VRAdaptationConfig()
```

#### Properties

- `LIGHTING_PARAMS`: Lighting adaptation ranges
- `AUDIO_PARAMS`: Audio adaptation settings
- `NPC_BEHAVIOR`: NPC behavior parameters

## Streamlit Components

### Real-time Video Processing

```python
import streamlit as st
from vr_emotion_adaptation import EmotionRecognitionSystem

# Initialize system
system = EmotionRecognitionSystem()

# Create video placeholder
video_placeholder = st.empty()

# Process frames
while True:
    frame, emotion_data = system.process_frame()
    video_placeholder.image(frame, channels="BGR")
```

### Emotion Analytics Dashboard

```python
import plotly.graph_objects as go
import streamlit as st

def create_emotion_chart(probabilities):
    fig = go.Figure(data=[
        go.Bar(x=EMOTION_LABELS, y=probabilities)
    ])
    fig.update_layout(title="Real-time Emotion Probabilities")
    return fig

# Display chart
st.plotly_chart(create_emotion_chart(emotion_probs))
```

## Data Structures

### Emotion Detection Result

```python
{
    'emotion': str,           # Detected emotion label
    'confidence': float,      # Confidence score (0.0-1.0)
    'probabilities': list,    # Probability for each emotion class
    'timestamp': datetime,    # Detection timestamp
    'face_bbox': tuple,      # Face bounding box (x, y, w, h)
    'processing_time': float  # Processing time in milliseconds
}
```

### VR Environment Parameters

```python
{
    'lighting': {
        'brightness': float,      # 0.3-1.0
        'temperature': int,       # 2000-10000K
        'saturation': float,      # 0.5-1.5
        'contrast': float         # 0.5-2.0
    },
    'audio': {
        'background_volume': float,  # 0.4-0.9
        'music_tempo': float,        # 0.5-2.0
        'reverb_intensity': float,   # 0.2-0.8
        'ambient_sounds': str        # Sound profile name
    },
    'npc_behavior': {
        'friendliness': float,       # -1.0 to 1.0
        'approach_distance': float,  # 1.0-10.0 meters
        'dialogue_complexity': float, # 0.0-1.0
        'response_speed': float      # 0.5-2.0
    }
}
```

### NPC Dialogue Response

```python
{
    'dialogue': str,          # Generated dialogue text
    'personality': str,       # NPC personality type
    'dialogue_type': str,     # Type of dialogue (greeting, comfort, etc.)
    'emotion_context': str,   # Emotion that triggered the dialogue
    'confidence': float,      # Generation confidence
    'generation_time': float  # Time taken to generate (ms)
}
```

## Error Handling

### Common Exceptions

```python
from models.ollama_integration import (
    OllamaIntegrationError,
    ModelNotAvailableError,
    DialogueGenerationError
)

try:
    dialogue = ollama.generate_npc_dialogue('happy')
except ModelNotAvailableError:
    print("Ollama model not available")
except DialogueGenerationError as e:
    print(f"Dialogue generation failed: {e}")
```

### Camera and Audio Errors

```python
from vr_emotion_adaptation import CameraError, AudioError

try:
    system.start_processing()
except CameraError:
    print("Camera initialization failed")
except AudioError:
    print("Audio device not available")
```

## Performance Monitoring

### System Metrics

```python
metrics = system.get_performance_metrics()
# Returns:
{
    'fps': float,                    # Current processing FPS
    'avg_processing_time': float,    # Average processing time (ms)
    'memory_usage': float,           # Memory usage (MB)
    'gpu_utilization': float,        # GPU utilization (%)
    'emotion_accuracy': float,       # Model accuracy (%)
    'total_frames_processed': int    # Total frames processed
}
```

### Debugging and Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# System provides detailed logs
system = EmotionRecognitionSystem(debug=True)
```

## Integration Examples

### Unity VR Integration

```csharp
// Unity C# script example
using UnityEngine;
using System.Net.Http;
using Newtonsoft.Json;

public class EmotionVRAdapter : MonoBehaviour
{
    private HttpClient client = new HttpClient();
    
    async void Update()
    {
        var response = await client.GetAsync("http://localhost:8501/api/vr_params");
        var json = await response.Content.ReadAsStringAsync();
        var vrParams = JsonConvert.DeserializeObject<VRParameters>(json);
        
        // Apply parameters to VR environment
        ApplyLighting(vrParams.lighting);
        ApplyAudio(vrParams.audio);
        UpdateNPCs(vrParams.npc_behavior);
    }
}
```

### Custom Emotion Processing

```python
from models.emotion_models import FacialEmotionModel, AudioEmotionModel

# Custom emotion processing pipeline
class CustomEmotionProcessor:
    def __init__(self):
        self.facial_model = FacialEmotionModel()
        self.audio_model = AudioEmotionModel()
    
    def process_multimodal(self, frame, audio_chunk):
        # Process facial emotions
        facial_result = self.facial_model.predict(frame)
        
        # Process audio emotions
        audio_result = self.audio_model.predict(audio_chunk)
        
        # Custom fusion logic
        combined_result = self.fuse_emotions(facial_result, audio_result)
        
        return combined_result
```

## Testing and Validation

### Unit Tests

```python
import unittest
from models.ollama_integration import OllamaIntegration

class TestOllamaIntegration(unittest.TestCase):
    def setUp(self):
        self.ollama = OllamaIntegration()
    
    def test_model_availability(self):
        self.assertTrue(self.ollama.is_model_available())
    
    def test_dialogue_generation(self):
        dialogue = self.ollama.generate_npc_dialogue('happy')
        self.assertIsInstance(dialogue, str)
        self.assertGreater(len(dialogue), 0)

if __name__ == '__main__':
    unittest.main()
```

### Performance Benchmarks

```python
import time
from vr_emotion_adaptation import EmotionRecognitionSystem

def benchmark_processing_speed():
    system = EmotionRecognitionSystem()
    
    start_time = time.time()
    for i in range(100):
        emotion_data = system.process_single_frame()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    fps = 1.0 / avg_time
    
    print(f"Average processing time: {avg_time:.3f}s")
    print(f"Estimated FPS: {fps:.1f}")
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Production Configuration

```python
# production_config.py
PRODUCTION_CONFIG = {
    'processing_fps': 15,
    'enable_gpu': True,
    'log_level': 'INFO',
    'cache_models': True,
    'enable_metrics': True,
    'ollama_timeout': 5.0
}
```