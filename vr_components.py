"""
Decomposed VR Emotion Adaptation components
"""
import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional, Tuple, Dict, Any
import threading
import queue

import cv2
import numpy as np
import pyaudio
import torch

from config_unified import config


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


class EmotionResult:
    """Value object for emotion recognition results"""
    
    def __init__(self, emotion: str, confidence: float, probabilities: np.ndarray):
        if not isinstance(emotion, str):
            raise TypeError("Emotion must be a string")
        if not isinstance(confidence, (int, float)):
            raise TypeError("Confidence must be numeric")
        if not isinstance(probabilities, np.ndarray):
            raise TypeError("Probabilities must be numpy array")
        
        self.emotion = emotion
        self.confidence = confidence
        self.probabilities = probabilities.copy()
    
    @classmethod
    def neutral(cls) -> 'EmotionResult':
        """Create neutral emotion result"""
        return cls('neutral', 0.1, config.model.neutral_probs)
    
    def __str__(self):
        return f"EmotionResult(emotion={self.emotion}, confidence={self.confidence:.2f})"


class ProcessingStrategy(ABC):
    """Abstract base class for emotion processing strategies"""
    
    @abstractmethod
    def process(self, data: Any) -> EmotionResult:
        """Process input data and return emotion result"""
        pass


class FacialProcessingStrategy(ProcessingStrategy):
    """Strategy for facial emotion processing"""
    
    def __init__(self, classifier, device: torch.device):
        self.classifier = classifier
        self.device = device
        self.face_cascade = self._load_cascade()
    
    def _load_cascade(self) -> cv2.CascadeClassifier:
        """Load face detection cascade"""
        cascade_path = f"{config.paths.haarcascade_path}/haarcascade_frontalface_default.xml"
        return cv2.CascadeClassifier(cascade_path)
    
    def process(self, frame: np.ndarray) -> EmotionResult:
        """Process facial frame for emotion recognition"""
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray_frame, 
                config.processing.face_detection_scale,
                config.processing.face_detection_min_neighbors
            )
            
            if len(faces) == 0:
                return EmotionResult.neutral()
            
            # Process largest face
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            face_roi = gray_frame[y:y+h, x:x+w]
            
            # Preprocess
            resized_face = cv2.resize(face_roi, config.model.face_input_size)
            resized_face = cv2.equalizeHist(resized_face)
            
            # Create tensor and predict
            input_tensor = torch.FloatTensor(resized_face).unsqueeze(0).unsqueeze(0) / 255.0
            prediction = self.classifier.predict(input_tensor.to(self.device))
            
            return EmotionResult(
                prediction['emotion'],
                prediction['confidence'],
                prediction['probabilities']
            )
            
        except Exception as e:
            logging.error(f"Facial processing error: {e}")
            return EmotionResult.neutral()


class DialogueController:
    """Controls dialogue generation with proper timing and state management"""
    
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


class ProcessingOrchestrator:
    """Orchestrates the emotion processing pipeline"""
    
    def __init__(self, resource_manager: ResourceManager, 
                 facial_strategy: ProcessingStrategy,
                 dialogue_controller: DialogueController):
        self.resource_manager = resource_manager
        self.facial_strategy = facial_strategy
        self.dialogue_controller = dialogue_controller
        self._logger = logging.getLogger(__name__)
    
    def process_frame(self) -> Dict[str, Any]:
        """Process a single frame and return results"""
        results = {
            'facial_result': EmotionResult.neutral(),
            'audio_result': EmotionResult.neutral(),
            'combined_result': EmotionResult.neutral(),
            'dialogue': '',
            'frame_processed': False
        }
        
        # Process video if available
        if self.resource_manager.video_capture:
            ret, frame = self.resource_manager.video_capture.read()
            if ret and frame is not None:
                results['facial_result'] = self.facial_strategy.process(frame)
                results['frame_processed'] = True
        
        # Determine dominant emotion (simplified for example)
        dominant_result = results['facial_result']
        
        # Generate dialogue if needed
        if self.dialogue_controller.should_generate_dialogue(
            dominant_result.emotion, dominant_result.confidence
        ):
            results['dialogue'] = f"I sense you're feeling {dominant_result.emotion}"
            self.dialogue_controller.update_dialogue_state(dominant_result.emotion)
        
        return results