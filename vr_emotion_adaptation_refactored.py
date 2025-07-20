"""
Refactored VR Emotion Adaptation System
Addresses god class anti-pattern and improves maintainability
"""

import logging
import os
import time
import queue
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from collections import deque

import cv2
import librosa
import numpy as np
import pyaudio
import torch

from config_unified import config
from models.emotion_models import EmotionClassifier
from vr_adaptation.environment_controller import EnvironmentController


@dataclass
class EmotionResult:
    """Value object for emotion recognition results"""
    emotion: str
    confidence: float
    probabilities: np.ndarray
    
    @classmethod
    def neutral(cls) -> 'EmotionResult':
        """Create neutral emotion result"""
        return cls('neutral', 0.1, config.model.neutral_probs.copy())
    
    def __post_init__(self):
        """Validate data after initialization"""
        if not isinstance(self.emotion, str):
            raise TypeError("Emotion must be a string")
        if not isinstance(self.confidence, (int, float)):
            raise TypeError("Confidence must be numeric")
        if not isinstance(self.probabilities, np.ndarray):
            raise TypeError("Probabilities must be numpy array")


class ResourceManager:
    """Manages hardware resources with proper lifecycle management"""
    
    def __init__(self):
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.audio_stream: Optional[pyaudio.Stream] = None
        self.pyaudio_instance: Optional[pyaudio.PyAudio] = None
        self._logger = logging.getLogger(__name__)
    
    @contextmanager
    def managed_resources(self):
        """Context manager for guaranteed resource cleanup"""
        try:
            yield self
        finally:
            self.release_all()
    
    def setup_video_capture(self) -> bool:
        """Setup video capture with comprehensive error handling"""
        try:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                self._logger.error("Could not open webcam")
                return False
            
            # Validate with test frame
            ret, frame = self.video_capture.read()
            if not ret or frame is None:
                self._logger.error("Could not read frame from webcam")
                self.release_video()
                return False
            
            self._logger.info(f"Webcam initialized: {frame.shape}")
            return True
            
        except Exception as e:
            self._logger.error(f"Video setup failed: {e}")
            self.release_video()
            return False
    
    def setup_audio_capture(self) -> bool:
        """Setup audio capture with comprehensive error handling"""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=config.processing.sample_rate,
                input=True,
                frames_per_buffer=config.processing.audio_chunk_size
            )
            self._logger.info("Audio stream initialized")
            return True
            
        except Exception as e:
            self._logger.error(f"Audio setup failed: {e}")
            self.release_audio()
            return False
    
    def release_video(self):
        """Safely release video resources"""
        if self.video_capture:
            try:
                self.video_capture.release()
                self.video_capture = None
            except Exception as e:
                self._logger.warning(f"Video release error: {e}")
    
    def release_audio(self):
        """Safely release audio resources"""
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
            except Exception as e:
                self._logger.warning(f"Audio stream release error: {e}")
        
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
            except Exception as e:
                self._logger.warning(f"PyAudio terminate error: {e}")
    
    def release_all(self):
        """Release all resources"""
        self.release_video()
        self.release_audio()
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            self._logger.warning(f"OpenCV cleanup error: {e}")


class EmotionProcessor:
    """Handles emotion processing logic with separation of concerns"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self._logger = logging.getLogger(__name__)
        self.face_cascade = self._load_face_cascade()
    
    def _load_face_cascade(self) -> cv2.CascadeClassifier:
        """Load face cascade classifier once"""
        cascade_path = os.path.join(config.paths.haarcascade_path, 'haarcascade_frontalface_default.xml')
        return cv2.CascadeClassifier(cascade_path)
    
    def process_facial_frame(self, frame: np.ndarray, classifier: EmotionClassifier) -> Tuple[EmotionResult, Optional[torch.Tensor]]:
        """Process facial emotion with improved error handling"""
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray_frame, 
                config.processing.face_detection_scale,
                config.processing.face_detection_min_neighbors
            )
            
            if len(faces) == 0:
                return EmotionResult.neutral(), None
            
            # Use largest detected face
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            face_roi = gray_frame[y:y+h, x:x+w]
            
            # Preprocess face
            resized_face = cv2.resize(face_roi, config.model.face_input_size)
            resized_face = cv2.equalizeHist(resized_face)
            
            # Create tensor and predict
            input_tensor = torch.FloatTensor(resized_face).unsqueeze(0).unsqueeze(0) / 255.0
            prediction = classifier.predict(input_tensor.to(self.device))
            
            result = EmotionResult(
                prediction['emotion'],
                prediction['confidence'],
                prediction['probabilities']
            )
            
            return result, input_tensor
            
        except Exception as e:
            self._logger.error(f"Facial processing error: {e}")
            return EmotionResult.neutral(), None
    
    def process_audio_chunk(self, audio_data: bytes, classifier: EmotionClassifier) -> Tuple[EmotionResult, Optional[torch.Tensor]]:
        """Process audio emotion with improved error handling"""
        try:
            # Convert audio data
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio_np, 
                sr=config.processing.sample_rate, 
                n_mfcc=config.model.mfcc_features
            )
            
            # Normalize length
            target_length = config.model.mfcc_target_length
            if mfcc.shape[1] < target_length:
                mfcc = np.pad(mfcc, ((0, 0), (0, target_length - mfcc.shape[1])), 'constant')
            else:
                mfcc = mfcc[:, :target_length]
            
            # Create tensor and predict
            mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).permute(0, 2, 1)
            prediction = classifier.predict(mfcc_tensor.to(self.device))
            
            result = EmotionResult(
                prediction['emotion'],
                prediction['confidence'],
                prediction['probabilities']
            )
            
            return result, mfcc_tensor
            
        except Exception as e:
            self._logger.error(f"Audio processing error: {e}")
            return EmotionResult.neutral(), None


class TemporalSmoother:
    """Handles temporal smoothing of emotion predictions"""
    
    def __init__(self, window_size: int = None):
        self.window_size = window_size or config.processing.smoothing_window
        self.emotion_history = deque(maxlen=self.window_size)
    
    def smooth(self, probabilities: np.ndarray) -> EmotionResult:
        """Apply temporal smoothing to probabilities"""
        self.emotion_history.append(probabilities)
        
        if len(self.emotion_history) == 0:
            return EmotionResult.neutral()
        
        # Average probabilities over window
        smoothed_probs = np.mean(list(self.emotion_history), axis=0)
        
        # Get dominant emotion
        dominant_idx = np.argmax(smoothed_probs)
        emotion_labels = config.model.emotion_labels
        dominant_emotion = emotion_labels[dominant_idx]
        confidence = smoothed_probs[dominant_idx]
        
        return EmotionResult(dominant_emotion, confidence, smoothed_probs)


class DialogueController:
    """Controls dialogue generation with proper timing"""
    
    def __init__(self, cooldown_seconds: float = None):
        self.cooldown_seconds = cooldown_seconds or config.processing.dialogue_cooldown
        self.last_dialogue_time = 0.0
        self.last_emotion = None
        self._logger = logging.getLogger(__name__)
    
    def should_generate_dialogue(self, emotion: str, confidence: float) -> bool:
        """Determine if dialogue should be generated based on rules"""
        current_time = time.time()
        
        # Always allow first dialogue
        if self.last_dialogue_time == 0:
            return True
        
        # Check cooldown
        if current_time - self.last_dialogue_time < self.cooldown_seconds:
            return False
        
        # Generate if emotion changed or high confidence
        emotion_changed = emotion != self.last_emotion
        high_confidence = confidence > config.processing.confidence_threshold
        
        return emotion_changed or high_confidence
    
    def update_dialogue_state(self, emotion: str):
        """Update dialogue generation state"""
        self.last_dialogue_time = time.time()
        self.last_emotion = emotion


class ThreadSafeCounter:
    """Thread-safe counter for frame counting"""
    
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self) -> int:
        """Increment counter and return new value"""
        with self._lock:
            self._value += 1
            return self._value
    
    @property
    def value(self) -> int:
        """Get current value"""
        with self._lock:
            return self._value


class VREmotionAdaptation:
    """Refactored VR Emotion Adaptation with improved architecture"""
    
    def __init__(self, facial_model=None, audio_model=None, fusion_model=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._logger = self._setup_logging()
        self._logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.resource_manager = ResourceManager()
        self.emotion_processor = EmotionProcessor(self.device)
        self.temporal_smoother = TemporalSmoother()
        self.dialogue_controller = DialogueController()
        self.frame_counter = ThreadSafeCounter()
        
        # Initialize classifiers
        self._initialize_classifiers(facial_model, audio_model, fusion_model)
        
        # Initialize environment controller
        self.env_controller = EnvironmentController()
        if hasattr(self.env_controller, 'ollama_client'):
            self.env_controller.ollama_client.model = "deepseek-r1"
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/backend.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _initialize_classifiers(self, facial_model, audio_model, fusion_model):
        """Initialize emotion classifiers"""
        self.facial_classifier = EmotionClassifier('facial', str(self.device), model=facial_model)
        self.audio_classifier = EmotionClassifier('audio', str(self.device), model=audio_model)
        self.multimodal_classifier = EmotionClassifier('multimodal', str(self.device), model=fusion_model)
        
        if not all([facial_model, audio_model, fusion_model]):
            self._create_dummy_weights()
            self._load_model_weights()
    
    def _create_dummy_weights(self):
        """Create dummy weights if they don't exist"""
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        weight_paths = [
            config.model.facial_model_weights,
            config.model.audio_model_weights,
            config.model.multimodal_model_weights
        ]
        
        classifiers = [
            self.facial_classifier,
            self.audio_classifier,
            self.multimodal_classifier
        ]
        
        for path, classifier in zip(weight_paths, classifiers):
            if not os.path.exists(path):
                torch.save(classifier.model.state_dict(), path)
                self._logger.info(f"Created dummy weights at {path}")
    
    def _load_model_weights(self):
        """Load model weights"""
        try:
            self.facial_classifier.load_weights(config.model.facial_model_weights)
            self.audio_classifier.load_weights(config.model.audio_model_weights)
            self.multimodal_classifier.load_weights(config.model.multimodal_model_weights)
        except Exception as e:
            self._logger.error(f"Failed to load model weights: {e}")
    
    def setup_realtime_capture(self) -> Tuple[bool, str]:
        """Setup real-time capture with improved error handling"""
        video_ok = self.resource_manager.setup_video_capture()
        audio_ok = self.resource_manager.setup_audio_capture()
        
        status_msg = f"Webcam: {'✅' if video_ok else '❌'}, Audio: {'✅' if audio_ok else '❌'}"
        
        if video_ok or audio_ok:
            self._logger.info("Real-time capture initialized successfully")
            return True, status_msg
        else:
            self._logger.warning("No capture devices available")
            return False, status_msg
    
    @contextmanager
    def _processing_context(self, stop_event, data_queue):
        """Context manager for processing with proper cleanup"""
        try:
            yield
        except Exception as e:
            self._logger.error(f"Processing error: {e}")
            try:
                data_queue.put_nowait({'type': 'error', 'message': str(e)})
            except queue.Full:
                pass
        finally:
            self.resource_manager.release_all()
    
    def run_realtime_adaptation(self, stop_event: threading.Event, data_queue: queue.Queue):
        """Main processing loop with improved architecture"""
        with self._processing_context(stop_event, data_queue):
            if not (self.resource_manager.video_capture or self.resource_manager.audio_stream):
                data_queue.put_nowait({
                    'type': 'error', 
                    'message': "No capture devices available"
                })
                return
            
            self._send_status_message(data_queue)
            
            while not stop_event.is_set():
                try:
                    self._process_single_frame(data_queue)
                    time.sleep(config.processing.processing_delay or 0.05)
                except Exception as e:
                    self._logger.error(f"Frame processing error: {e}")
                    break
            
            data_queue.put_nowait({'type': 'info', 'message': "Processing stopped"})
    
    def _send_status_message(self, data_queue: queue.Queue):
        """Send initial status message"""
        if self.resource_manager.video_capture and self.resource_manager.audio_stream:
            message = "Starting multimodal emotion detection"
        elif self.resource_manager.video_capture:
            message = "Starting video-only emotion detection"
        else:
            message = "Starting audio-only emotion detection"
        
        try:
            data_queue.put_nowait({'type': 'info', 'message': message})
        except queue.Full:
            pass
    
    def _process_single_frame(self, data_queue: queue.Queue):
        """Process a single frame of data"""
        # Process facial emotion
        facial_result = EmotionResult.neutral()
        facial_tensor = None
        
        if self.resource_manager.video_capture:
            ret, frame = self.resource_manager.video_capture.read()
            if ret and frame is not None:
                facial_result, facial_tensor = self.emotion_processor.process_facial_frame(
                    frame, self.facial_classifier
                )
        
        # Process audio emotion
        audio_result = EmotionResult.neutral()
        audio_tensor = None
        
        if self.resource_manager.audio_stream:
            try:
                audio_data = self.resource_manager.audio_stream.read(
                    config.processing.audio_chunk_size, 
                    exception_on_overflow=False
                )
                audio_result, audio_tensor = self.emotion_processor.process_audio_chunk(
                    audio_data, self.audio_classifier
                )
            except Exception as e:
                self._logger.warning(f"Audio read error: {e}")
        
        # Multimodal fusion
        combined_result = self._fuse_emotions(facial_result, audio_result, facial_tensor, audio_tensor)
        
        # Temporal smoothing
        smoothed_result = self.temporal_smoother.smooth(combined_result.probabilities)
        
        # Update environment
        self.env_controller.update_emotion(smoothed_result.emotion, smoothed_result.confidence)
        
        # Generate dialogue if needed
        dialogue = self._generate_dialogue_if_needed(smoothed_result)
        
        # Send update
        self._send_frame_update(data_queue, facial_result, audio_result, combined_result, smoothed_result, dialogue)
    
    def _fuse_emotions(self, facial_result: EmotionResult, audio_result: EmotionResult, 
                      facial_tensor: Optional[torch.Tensor], audio_tensor: Optional[torch.Tensor]) -> EmotionResult:
        """Fuse facial and audio emotions"""
        if facial_tensor is not None and audio_tensor is not None:
            try:
                # Flatten tensors for fusion
                facial_flat = facial_tensor.view(facial_tensor.size(0), -1)
                audio_flat = audio_tensor.view(audio_tensor.size(0), -1)
                combined_input = torch.cat([facial_flat, audio_flat], dim=1)
                
                prediction = self.multimodal_classifier.predict(combined_input.to(self.device))
                return EmotionResult(
                    prediction['emotion'],
                    prediction['confidence'],
                    prediction['probabilities']
                )
            except Exception as e:
                self._logger.warning(f"Multimodal fusion failed: {e}")
        
        # Fallback to averaging
        avg_probs = (facial_result.probabilities + audio_result.probabilities) / 2
        dominant_idx = np.argmax(avg_probs)
        emotion = config.model.emotion_labels[dominant_idx]
        confidence = avg_probs[dominant_idx]
        
        return EmotionResult(emotion, confidence, avg_probs)
    
    def _generate_dialogue_if_needed(self, emotion_result: EmotionResult) -> str:
        """Generate dialogue if conditions are met"""
        if self.dialogue_controller.should_generate_dialogue(
            emotion_result.emotion, emotion_result.confidence
        ):
            dialogue = self.env_controller.generate_dialogue(emotion_result.emotion)
            self.dialogue_controller.update_dialogue_state(emotion_result.emotion)
            return dialogue
        return ""
    
    def _send_frame_update(self, data_queue: queue.Queue, facial_result: EmotionResult, 
                          audio_result: EmotionResult, combined_result: EmotionResult,
                          smoothed_result: EmotionResult, dialogue: str):
        """Send frame update to UI"""
        frame_count = self.frame_counter.increment()
        
        try:
            data_queue.put_nowait({
                'type': 'update',
                'facial_emotion_name': facial_result.emotion,
                'facial_probs': facial_result.probabilities,
                'audio_emotion_name': audio_result.emotion,
                'audio_probs': audio_result.probabilities,
                'combined_emotion_name': combined_result.emotion,
                'combined_probs': combined_result.probabilities,
                'dominant_emotion': smoothed_result.emotion,
                'dominant_confidence': smoothed_result.confidence,
                'dialogue': dialogue,
                'frame_count': frame_count
            })
        except queue.Full:
            self._logger.warning("Queue full, skipping update")
    
    def release_resources(self):
        """Release all resources"""
        self.resource_manager.release_all()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_resources()