# VR Emotion Adaptation System

A sophisticated real-time emotion-driven environmental adaptation system for Virtual Reality environments. This system analyzes user emotions through facial expressions and voice patterns, then dynamically adapts the VR environment's lighting, sound, and NPC behavior to create personalized, immersive experiences.

## 🚀 Features

- **Multi-modal Emotion Recognition**: Advanced fusion of facial expression and voice analysis
- **Real-time Processing**: Low-latency emotion detection with live video streaming
- **Dynamic VR Adaptation**: Intelligent adjustment of lighting, audio, and NPC behavior
- **AI-Powered NPCs**: Advanced Ollama integration for contextual dialogue generation
- **Interactive Web Interface**: Streamlit-based dashboard for real-time monitoring
- **Temporal Smoothing**: Sophisticated emotion prediction stabilization
- **Comprehensive Analytics**: Performance metrics, emotion history, and accuracy tracking
- **Modular Architecture**: Extensible design for custom VR integrations

## System Requirements

### Hardware

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **Camera**: Webcam for facial expression capture
- **Microphone**: Audio input for voice emotion analysis

### Software

- **OS**: Windows 10/11 with WSL2, or Linux
- **Python**: 3.8-3.12
- **CUDA**: 11.8+ (for GPU acceleration)
- **Ollama**: Local LLM for NPC dialogue generation

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Conference
```

### 2. Set Up WSL2 Environment (Windows)

```bash
# Install WSL2 if not already installed
wsl --install

# Install CUDA and cuDNN in WSL2
# Follow NVIDIA's WSL2 CUDA setup guide
```

### 3. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/WSL2
# or
venv\Scripts\activate     # Windows
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install System Dependencies (WSL2/Linux)

```bash
sudo apt update
sudo apt install -y python3-dev cmake build-essential portaudio19-dev
```

### 6. Install and Configure Ollama

```bash
# Download and install Ollama from https://ollama.ai
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull the DeepSeek model (recommended for dialogue generation)
ollama pull deepseek-r1:latest

# Alternative models
ollama pull qwen2.5:latest
ollama pull llama3
```

### 7. Test Ollama Integration

```bash
# Activate virtual environment
.\venv_py311_local\Scripts\Activate.ps1  # Windows
# or
source venv_py311_local/bin/activate     # Linux

# Test Ollama connectivity
python test_ollama.py
```

## 🎯 Usage

### 1. Quick Start - Web Interface

```bash
# Activate virtual environment
.\venv_py311_local\Scripts\Activate.ps1  # Windows
# or
source venv_py311_local/bin/activate     # Linux

# Launch the Streamlit web application
streamlit run app.py
```

The web interface provides:

- **Real-time Video Feed**: Live emotion detection overlay
- **Emotion Analytics**: Real-time emotion probabilities and confidence scores
- **VR Environment Controls**: Dynamic parameter visualization
- **NPC Dialogue Generation**: AI-powered contextual responses
- **Performance Metrics**: FPS, processing latency, and accuracy tracking

### 2. Training Custom Models (Optional)

```bash
python train_models.py
```

This will:

- Train facial emotion recognition model on FER2013 dataset
- Train audio emotion recognition model on custom audio dataset
- Generate comprehensive training plots and confusion matrices
- Save optimized models to `models/` directory
- Create evaluation reports with accuracy metrics

### 3. Advanced Configuration

Edit configuration files to customize behavior:

**`config.py`** - Core system settings:

- Model paths and architectures
- Processing parameters and thresholds
- Device selection (GPU/CPU)
- Logging and debug settings

**`config_unified.py`** - VR adaptation parameters:

- Environment adaptation rules
- Emotion-to-parameter mappings
- Temporal smoothing settings
- NPC behavior configurations

## 📁 Project Structure

```
Conference/
├── app.py                          # Streamlit web application (main entry)
├── main.py                         # Alternative CLI entry point
├── config.py                       # Core system configuration
├── config_unified.py               # VR adaptation parameters
├── train_models.py                 # Model training pipeline
├── test_ollama.py                  # Ollama connectivity testing
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── venv_py311_local/               # Python virtual environment
├── models/
│   ├── emotion_models.py           # Neural network architectures
│   ├── ollama_integration.py       # Advanced Ollama LLM integration
│   ├── context_aware_emotion.py    # Context-aware emotion processing
│   ├── advanced_fusion.py          # Multi-modal emotion fusion
│   ├── facial_emotion_model.pth    # Trained facial emotion model
│   └── audio_emotion_model.pth     # Trained audio emotion model
├── vr_adaptation/
│   └── environment_controller.py   # VR environment adaptation engine
├── vr_components.py                # VR system components
├── utils.py                        # Utility functions and helpers
├── data/
│   └── data_loader.py             # Dataset loading and preprocessing
├── archive/                        # FER2013 facial expression dataset
├── archive (1)/                    # Audio emotion dataset
├── logs/
│   └── backend.log                # System logs and debug info
├── output/                         # Generated plots and analysis
├── evaluation_results/             # Model evaluation metrics
└── tests/                          # Unit tests and diagnostics
    ├── test_audio.py
    ├── test_webcam.py
    └── diagnostic_test.py
```

## 🎭 Emotion Recognition & AI Integration

### Supported Emotions

The system recognizes 7 basic emotions with sophisticated environmental adaptations:

| Emotion      | Visual Adaptation                   | Audio Adaptation               | NPC Behavior                     |
| ------------ | ----------------------------------- | ------------------------------ | -------------------------------- |
| **Happy**    | Bright, warm lighting (3000K-4000K) | Upbeat music, cheerful sounds  | Friendly, celebratory dialogue   |
| **Sad**      | Dim, cool lighting (5000K-6500K)    | Melancholic sounds, soft music | Comforting, supportive responses |
| **Angry**    | High contrast, red-tinted lighting  | Intense music, dramatic sounds | Calming, understanding dialogue  |
| **Fear**     | Dynamic lighting, shadows           | Tense audio, protective sounds | Reassuring, protective responses |
| **Surprise** | Bright lighting, sudden changes     | Alert sounds, curious music    | Excited, curious dialogue        |
| **Disgust**  | Neutral lighting, muted colors      | Subdued audio, neutral tones   | Understanding, non-judgmental    |
| **Neutral**  | Balanced, natural environment       | Ambient background sounds      | Observant, engaging conversation |

### AI-Powered NPC Dialogue System

The system uses **Ollama** with advanced language models for contextual NPC interactions:

#### Supported Models

- **DeepSeek-R1** (Recommended): Advanced reasoning and dialogue generation
- **Qwen2.5**: Multilingual support and creative responses
- **Llama3**: General-purpose conversational AI

#### Dialogue Features

- **Context-Aware Responses**: NPCs understand emotional context
- **Personality Adaptation**: 22 different NPC personality types
- **Dynamic Dialogue Types**: Greetings, reactions, comfort, encouragement, etc.
- **Response Cleaning**: Removes verbose model outputs for VR-appropriate dialogue
- **Fallback System**: Template-based responses when AI is unavailable

#### NPC Personality Types

```python
# Available NPC personalities that adapt to user emotions
FRIENDLY, WISE, PLAYFUL, MYSTERIOUS, PROFESSIONAL, CARING,
ADVENTUROUS, PHILOSOPHICAL, HUMOROUS, SERIOUS, CELEBRATORY,
SUPPORTIVE, CALMING, REASSURING, PROTECTIVE, CURIOUS,
EXCITED, UNDERSTANDING, NEUTRAL, HELPFUL, OBSERVANT, ENGAGING
```

## VR Environment Adaptation

### Lighting Adaptation

- **Brightness**: 0.3-1.0 (dim to bright)
- **Temperature**: 2000K-10000K (warm to cool)
- **Saturation**: 0.5-1.5 (muted to vibrant)
- **Contrast**: 0.5-2.0 (low to high contrast)

### Audio Adaptation

- **Background Volume**: 0.4-0.9
- **Music Tempo**: 0.5-2.0 (slow to fast)
- **Reverb Intensity**: 0.2-0.8
- **Ambient Sounds**: Emotion-specific sound effects

### NPC Behavior Adaptation

- **Friendliness**: -1.0 to 1.0 (hostile to friendly)
- **Approach Distance**: 1.0-10.0 meters
- **Dialogue Complexity**: 0.0-1.0
- **Response Speed**: 0.5-2.0

## Performance Optimization

### GPU Acceleration

The system automatically detects and uses CUDA-capable GPUs for:

- Facial emotion recognition
- Audio emotion processing
- Multi-modal fusion

### Real-time Processing

- **Target FPS**: 10 FPS (configurable)
- **Processing Latency**: ~350ms total
- **Memory Usage**: ~2GB RAM

### Temporal Smoothing

- **Smoothing Window**: 5 frames (configurable)
- **Confidence Threshold**: 0.6 (configurable)
- **Stability**: Reduces emotion prediction jitter

## 🖥️ Streamlit Web Interface

The Streamlit interface provides a comprehensive dashboard for real-time emotion monitoring and VR adaptation:

### Main Dashboard Features

#### 1. Real-time Video Feed

- Live webcam stream with emotion detection overlay
- Confidence scores displayed on detected faces
- Real-time emotion classification labels

#### 2. Emotion Analytics Panel

- **Current Emotion**: Dominant emotion with confidence percentage
- **Emotion Probabilities**: Real-time bar chart of all 7 emotions
- **Emotion History**: Timeline graph showing emotion changes over time
- **Accuracy Metrics**: Model performance statistics

#### 3. VR Environment Controls

- **Lighting Parameters**: Brightness, temperature, saturation, contrast
- **Audio Settings**: Volume, tempo, reverb, ambient sound selection
- **NPC Behavior**: Friendliness, approach distance, dialogue complexity

#### 4. AI Dialogue Generation

- **Live NPC Responses**: Real-time dialogue generation based on detected emotions
- **Dialogue History**: Log of all generated NPC interactions
- **Model Status**: Ollama connection and model availability indicators

#### 5. System Performance

- **Processing FPS**: Real-time frame rate monitoring
- **Latency Metrics**: Processing time breakdown
- **Resource Usage**: CPU/GPU utilization and memory consumption

### Interface Controls

```python
# Key Streamlit components used in the interface
st.video()           # Live video stream display
st.plotly_chart()    # Interactive emotion probability charts
st.metrics()         # Real-time performance indicators
st.sidebar()         # Configuration controls
st.expander()        # Collapsible sections for detailed info
```

## 🔧 Troubleshooting

### Common Issues

1. **Camera Not Found**

   ```bash
   # Check available cameras
   python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).read()[0]])"
   ```

   - Check camera permissions in system settings
   - Ensure camera is not used by other applications
   - Try different camera index in configuration

2. **Audio Input Issues**

   ```bash
   # Test audio devices
   python -c "import pyaudio; p=pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)}') for i in range(p.get_device_count())]"
   ```

   - Check microphone permissions
   - Verify audio device is working
   - Install PortAudio: `sudo apt install portaudio19-dev`

3. **CUDA/GPU Issues**

   ```bash
   # Check CUDA availability
   nvidia-smi
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"CPU\"}')"
   ```

   - Install CUDA-compatible PyTorch version
   - Update GPU drivers
   - Check CUDA toolkit installation

4. **Ollama Connection Issues**

   ```bash
   # Test Ollama connectivity
   python test_ollama.py

   # Manual Ollama commands
   ollama serve                    # Start Ollama service
   ollama list                     # Check available models
   ollama pull deepseek-r1:latest  # Download model
   ```

5. **Streamlit Issues**

   ```bash
   # Clear Streamlit cache
   streamlit cache clear

   # Run with specific port
   streamlit run app.py --server.port 8502
   ```

### Performance Issues

1. **Low FPS (< 5 FPS)**

   - Reduce video resolution in `config.py`
   - Disable temporal smoothing temporarily
   - Switch to CPU-only mode for testing
   - Close other resource-intensive applications

2. **High Memory Usage (> 4GB)**

   - Reduce batch size in model configuration
   - Clear emotion history periodically
   - Use smaller model architectures
   - Enable garbage collection in processing loop

3. **Dialogue Generation Delays**
   - Check Ollama model size (smaller models = faster responses)
   - Reduce dialogue prompt complexity
   - Enable dialogue caching
   - Use fallback templates for immediate responses

### Debug Mode

Enable comprehensive debugging:

```python
# In config.py
DEBUG_MODE = True
VERBOSE_LOGGING = True
SAVE_DEBUG_FRAMES = True
```

This enables:

- Detailed console output
- Frame-by-frame processing logs
- Performance profiling
- Error stack traces

## Development

### Adding New Emotions

1. Update `emotion_labels` in `config.py`
2. Retrain models with new emotion classes
3. Add emotion configurations in `environment_controller.py`

### Custom VR Integration

1. Modify `get_unity_parameters()` in `environment_controller.py`
2. Implement your VR engine's parameter format
3. Add custom adaptation logic

### Extending AI Integration

1. Add new Ollama models in `ollama_integration.py`
2. Implement custom prompt templates
3. Add new generation capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this system in your research, please cite:

```bibtex
@article{vr_emotion_adaptation_2024,
  title={Emotion-Driven Environmental Adaptation in VR Worlds: A Multi-Modal Approach},
  author={[Your Name]},
  journal={[Conference/Journal Name]},
  year={2024}
}
```

## Acknowledgments

- FER2013 dataset for facial emotion recognition
- Audio emotion dataset contributors
- Ollama team for local LLM integration
- OpenCV and PyTorch communities

## 📚 Documentation

### Complete Documentation Suite

- **[Installation Guide](INSTALLATION_GUIDE.md)** - Comprehensive setup instructions for all platforms
- **[API Documentation](API_DOCUMENTATION.md)** - Complete API reference and integration examples
- **[Code Analysis & Improvements](CODE_ANALYSIS_IMPROVEMENTS.md)** - Technical analysis and optimization strategies

### Quick Reference

| Document                        | Description                           | Target Audience                   |
| ------------------------------- | ------------------------------------- | --------------------------------- |
| `README.md`                     | Project overview and quick start      | All users                         |
| `INSTALLATION_GUIDE.md`         | Detailed setup instructions           | New users, system administrators  |
| `API_DOCUMENTATION.md`          | API reference and examples            | Developers, integrators           |
| `CODE_ANALYSIS_IMPROVEMENTS.md` | Technical deep-dive and optimizations | Advanced developers, contributors |

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow coding standards** outlined in `CODE_ANALYSIS_IMPROVEMENTS.md`
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/vr-emotion-adaptation.git
cd vr-emotion-adaptation

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run code quality checks
flake8 .
black .
mypy .
```

## 📞 Support & Community

### Getting Help

1. **📖 Documentation**: Check our comprehensive documentation suite
2. **🐛 Issues**: Report bugs on [GitHub Issues](https://github.com/your-username/vr-emotion-adaptation/issues)
3. **💬 Discussions**: Join conversations on [GitHub Discussions](https://github.com/your-username/vr-emotion-adaptation/discussions)
4. **📧 Email**: Contact the maintainers at [email@example.com]

### Troubleshooting Resources

- **System Diagnostics**: Run `python diagnostic_test.py`
- **Log Analysis**: Check `logs/backend.log` for detailed error messages
- **Performance Monitoring**: Use built-in metrics dashboard
- **Community Wiki**: Browse community-contributed solutions

## 🎯 Roadmap & Future Development

### Upcoming Features (v2.0)

- **🧠 Advanced AI Models**: Integration with GPT-4 and Claude for enhanced NPC dialogue
- **🎮 VR Platform Support**: Native Unity and Unreal Engine plugins
- **📱 Mobile Support**: iOS and Android emotion recognition apps
- **🌐 Cloud Deployment**: Scalable cloud-based processing
- **🔒 Privacy Features**: On-device processing and data encryption

### Long-term Vision (v3.0+)

- **🤖 Autonomous NPCs**: Self-learning NPCs that adapt to individual users
- **🌍 Metaverse Integration**: Cross-platform VR world compatibility
- **🧬 Biometric Integration**: Heart rate, EEG, and other physiological signals
- **🎨 Procedural Content**: AI-generated environments based on emotions

## 🏆 Acknowledgments

### Research & Development

- **Emotion Recognition**: Based on FER2013 and custom audio emotion datasets
- **AI Integration**: Powered by Ollama and open-source language models
- **VR Adaptation**: Inspired by research in affective computing and immersive experiences

### Open Source Libraries

- **PyTorch**: Deep learning framework for emotion recognition models
- **OpenCV**: Computer vision library for facial detection and processing
- **Streamlit**: Web application framework for the interactive dashboard
- **Ollama**: Local language model inference for NPC dialogue generation

### Community Contributors

- Special thanks to all contributors who have helped improve this project
- Beta testers who provided valuable feedback and bug reports
- Researchers who shared datasets and methodologies

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation

If you use this system in your research or commercial projects, please cite:

```bibtex
@software{vr_emotion_adaptation_2024,
  title={VR Emotion Adaptation System: Real-time Emotion-Driven Environmental Adaptation},
  author={[Your Name]},
  year={2024},
  url={https://github.com/your-username/vr-emotion-adaptation},
  version={1.0.0}
}
```

## 🚀 Quick Start Summary

1. **Install**: Follow the [Installation Guide](INSTALLATION_GUIDE.md)
2. **Configure**: Set up Ollama and test connectivity with `python test_ollama.py`
3. **Run**: Launch with `streamlit run app.py`
4. **Explore**: Use the web interface to monitor real-time emotion detection
5. **Integrate**: Follow [API Documentation](API_DOCUMENTATION.md) for VR integration

**Ready to transform your VR experiences with emotion-driven adaptation!** 🎉

## PyTorch Usage and GPU/CPU Selection

This project uses PyTorch for all deep learning tasks. The system will automatically use your GPU (CUDA) if available, or fall back to CPU if not. You can check and select the device in your scripts as follows:

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Example: Move model and data to GPU
model = MyModel().to(device)
inputs = inputs.to(device)
outputs = model(inputs)
```

### Minimal PyTorch Training Loop Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Example model
def MyModel():
    return nn.Sequential(
        nn.Linear(10, 2)
    )

# Data
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16)

# Model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(10):
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```
