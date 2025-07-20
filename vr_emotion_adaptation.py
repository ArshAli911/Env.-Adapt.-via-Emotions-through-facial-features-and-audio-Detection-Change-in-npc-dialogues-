import logging
import os
import time
import queue
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import librosa
import numpy as np
import pyaudio
import torch

from data.data_loader import FacialExpressionDataset, AudioEmotionDataset
from models.emotion_models import FacialEmotionCNN, AudioEmotionLSTM, MultiModalEmotionFusion, EmotionClassifier
from vr_adaptation.environment_controller import EnvironmentController, EmotionType
from config_unified import config
# Logging configuration moved to a proper setup function
def setup_logging():
    """Configure logging for the VR emotion adaptation system."""
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

from config_unified import config

class VREmotionAdaptation:
    def __init__(self, facial_model=None, audio_model=None, fusion_model=None):
        self._logger = setup_logging()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            self._logger.warning("CUDA is not available. The application will run on CPU.")
        self._logger.info(f"Using device: {self.device}")
        
        # Use cached models if provided, otherwise load as before
        self.facial_classifier = EmotionClassifier('facial', str(self.device), model=facial_model)
        self.audio_classifier = EmotionClassifier('audio', str(self.device), model=audio_model)
        self.multimodal_classifier = EmotionClassifier('multimodal', str(self.device), model=fusion_model)
        
        if not (facial_model and audio_model and fusion_model):
            # Load dummy weights for demonstration (replace with actual trained weights)
            self._create_dummy_weights()
            self.facial_classifier.load_weights(config.model.facial_model_weights)
            self.audio_classifier.load_weights(config.model.audio_model_weights)
            self.multimodal_classifier.load_weights(config.model.multimodal_model_weights)
        
        # Initialize environment controller with DeepSeek model
        self.env_controller = EnvironmentController()
        if hasattr(self.env_controller, 'ollama_client'):
            self.env_controller.ollama_client.model = "deepseek-r1"
        self.audio_stream = None
        self.video_capture = None

        # Data collection for plotting (these will be accessed by the worker thread)
        self.timestamps = deque(maxlen=100)
        self.facial_probs_history = deque(maxlen=100)
        self.audio_probs_history = deque(maxlen=100)
        self.combined_probs_history = deque(maxlen=100)
        self.dominant_emotion_history = deque(maxlen=100)
        # Temporal smoothing for emotion stability
        self.emotion_history = deque(maxlen=5) # Number of frames to average
        self.smoothing_window = 5  # Number of frames to average
        self.confidence_threshold = 0.6  # Minimum confidence for emotion change
        # Dialogue generation control
        self.last_dialogue_time = 0.0
        self.dialogue_cooldown = 5 
        self.last_dialogue_emotion = None
        
        # Accuracy tracking
        self.facial_accuracies = []
        self.audio_accuracies = []
        self.combined_accuracies = []
        self.frame_count = 0
        self.accuracy_update_interval = 30  # Update accuracy every 30 frames
    
    def _cleanup_audio_on_error(self):
        """Clean up audio resources when an error occurs during setup."""
        if hasattr(self, 'audio_stream') and self.audio_stream:
            try:
                self.audio_stream.close()
            except Exception:
                pass  # Ignore cleanup errors
            self.audio_stream = None
        
        if hasattr(self, 'pyaudio_instance') and self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except Exception:
                pass  # Ignore cleanup errors
            self.pyaudio_instance = None
    
    def _create_dummy_weights(self):
        """Create dummy weights for models if they don't exist."""
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        if not os.path.exists(config.model.facial_model_weights):
            torch.save(self.facial_classifier.model.state_dict(), config.model.facial_model_weights)
            self._logger.info(f"Created dummy weights for facial model at {config.model.facial_model_weights}")
            
        if not os.path.exists(config.model.audio_model_weights):
            torch.save(self.audio_classifier.model.state_dict(), config.model.audio_model_weights)
            self._logger.info(f"Created dummy weights for audio model at {config.model.audio_model_weights}")
            
        if not os.path.exists(config.model.multimodal_model_weights):
            torch.save(self.multimodal_classifier.model.state_dict(), config.model.multimodal_model_weights)
            self._logger.info(f"Created dummy weights for multimodal model at {config.model.multimodal_model_weights}")

    def setup_realtime_capture(self):
        self._logger.debug("setup_realtime_capture called.")
        
        # Setup webcam
        self._logger.debug("Attempting to open webcam...")
        try:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                self._logger.error("Could not open webcam.")
                self.video_capture = None
            else:
                # Test read a frame to make sure it's working
                ret, frame = self.video_capture.read()
                if not ret or frame is None:
                    self._logger.error("Could not read frame from webcam.")
                    self.video_capture.release()
                    self.video_capture = None
                else:
                    self._logger.debug(f"Webcam successfully opened. Frame shape: {frame.shape}")
        except Exception as e:
            self._logger.error(f"Exception when opening webcam: {e}")
            self.video_capture = None
        
        # Setup audio capture
        self._logger.debug("Attempting to open audio stream...")
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=config.processing.sample_rate,
                input=True,
                frames_per_buffer=config.processing.audio_chunk_size
            )
            self._logger.debug("Audio stream successfully opened.")
        except pyaudio.PortAudioError as e:
            self._logger.error(f"Audio device error: {e}")
            self.audio_stream = None
            self._cleanup_audio_on_error()
        except OSError as e:
            self._logger.error(f"Audio system error: {e}")
            self.audio_stream = None
            self._cleanup_audio_on_error()
        except Exception as e:
            self._logger.error(f"Unexpected audio error: {e}")
            self.audio_stream = None
            self._cleanup_audio_on_error()
            raise  # Re-raise unexpected errors

        # Report status
        webcam_status = "✅ Available" if self.video_capture else "❌ Not Available"
        audio_status = "✅ Available" if self.audio_stream else "❌ Not Available"
        
        if self.video_capture and self.audio_stream:
            self._logger.info("Real-time audio and video capture initialized successfully.")
            return True, f"Webcam: {webcam_status}, Audio: {audio_status}"
        else:
            self._logger.warning("Real-time audio and/or video capture failed to initialize fully.")
            return False, f"Webcam: {webcam_status}, Audio: {audio_status}"
    
    def process_facial_frame(self, frame):
        """Process a single video frame for facial emotion recognition"""
        try:
            # Convert to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Face detection using Haar Cascade (more robust than just resizing)
            face_cascade = cv2.CascadeClassifier(os.path.join(config.paths.haarcascade_path, 'haarcascade_frontalface_default.xml'))
            faces = face_cascade.detectMultiScale(gray_frame, config.processing.face_detection_scale, config.processing.face_detection_min_neighbors)
            
            if len(faces) == 0:
                # No face detected, return neutral emotion with low confidence and None for tensor
                return {
                    'emotion': 'neutral',
                    'confidence': 0.1,
                    'probabilities': config.model.neutral_probs.copy()
                }, None
        except Exception as e:
            self._logger.error(f"Error in face detection: {e}")
            return {
                'emotion': 'neutral',
                'confidence': 0.1,
                'probabilities': config.model.neutral_probs.copy()
            }, None
        
        # Use the largest face detected
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        face_roi = gray_frame[y:y+h, x:x+w]
        
        # Resize to model input size
        resized_face = cv2.resize(face_roi, config.model.face_input_size)
        
        # Apply histogram equalization for better contrast
        resized_face = cv2.equalizeHist(resized_face)
        
        # Normalize and prepare tensor
        input_tensor = torch.FloatTensor(resized_face).unsqueeze(0).unsqueeze(0) / 255.0
        
        facial_prediction = self.facial_classifier.predict(input_tensor.to(self.device))
        return facial_prediction, input_tensor
    
    def process_audio_chunk(self, audio_data):
        """Process a chunk of audio data for voice emotion recognition"""
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Extract MFCC features (ensure consistent length for LSTM input)
        mfcc = librosa.feature.mfcc(y=audio_np, sr=config.processing.sample_rate, n_mfcc=config.model.mfcc_features)
        
        # Pad or truncate to a fixed length
        target_length = config.model.mfcc_target_length
        if mfcc.shape[1] < target_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, target_length - mfcc.shape[1])), 'constant')
        else:
            mfcc = mfcc[:, :target_length]
            
        mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0)  # Add batch dimension
        
        audio_prediction = self.audio_classifier.predict(mfcc_tensor.permute(0, 2, 1).to(self.device))
        return audio_prediction, mfcc_tensor.permute(0, 2, 1) # Return permuted tensor for multimodal input
        
    def run_realtime_adaptation(self, stop_event, data_queue):
        self._logger.debug("run_realtime_adaptation thread started.")
        
        # Allow processing even if only one input is available
        if not self.video_capture and not self.audio_stream:
            try:
                data_queue.put({'type': 'error', 'message': "Neither webcam nor audio capture is available. Please check your devices."})
            except:
                pass  # Queue might be full, ignore
            return
        
        try:
            if not self.video_capture:
                data_queue.put({'type': 'info', 'message': "Webcam not available. Running with audio-only emotion detection."})
            elif not self.audio_stream:
                data_queue.put({'type': 'info', 'message': "Audio not available. Running with video-only emotion detection."})
            else:
                data_queue.put({'type': 'info', 'message': "Starting real-time VR emotion adaptation with both video and audio."})
        except:
            pass  # Queue might be full, ignore initial messages
            
        # Initialize tracking variables
        self.last_emotion = None
        self.last_confidence = 0.0
        self.frame_count = 0
        
        while not stop_event.is_set():
            facial_input_tensor = None  # Initialize to None at the start of each loop
            audio_mfcc_tensor = None    # Initialize to None at the start of each loop
            try:
                # Initialize default results
                facial_emotion_result = {
                    'emotion': 'neutral',
                    'confidence': 0.1,
                    'probabilities': np.array([0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1])
                }
                
                # Capture and process video frame if available
                if self.video_capture is not None:
                    ret, frame = self.video_capture.read()
                    self._logger.debug(f"Webcam frame read: ret={ret}")
                    if ret and frame is not None:
                        try:
                            facial_emotion_result, facial_input_tensor = self.process_facial_frame(frame)
                        except Exception as e:
                            self._logger.error(f"Facial processing error: {e}")
                            facial_input_tensor = None
                    else:
                        self._logger.warning("Failed to read frame from webcam")
                        facial_input_tensor = None
                else:
                    facial_input_tensor = None
                
                self._logger.debug(f"Facial Emotion Result: {facial_emotion_result}")
                
                # Initialize default audio result
                audio_emotion_result = {
                    'emotion': 'neutral',
                    'confidence': 0.1,
                    'probabilities': np.array([0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1])
                }
                
                # Process audio if available
                if self.audio_stream:
                    try:
                        audio_data = self.audio_stream.read(config.processing.audio_chunk_size, exception_on_overflow=False)
                        audio_emotion_result, audio_mfcc_tensor = self.process_audio_chunk(audio_data)
                    except Exception as e:
                        print(f"Audio processing error: {e}")
                        audio_mfcc_tensor = None
                else:
                    audio_mfcc_tensor = None
                
                print(f"[DEBUG] Audio Emotion Result: {audio_emotion_result}") # Added debug print
                
                # Multimodal fusion (if both inputs are available)
                if facial_input_tensor is not None and audio_mfcc_tensor is not None:
                    try:
                        # Flatten both tensors to [batch, -1] before concatenation
                        facial_flat = facial_input_tensor.view(facial_input_tensor.size(0), -1)
                        audio_flat = audio_mfcc_tensor.view(audio_mfcc_tensor.size(0), -1)
                        combined_input = torch.cat([facial_flat, audio_flat], dim=1)
                        combined_emotion_result = self.multimodal_classifier.predict(combined_input.to(self.device))
                    except Exception as e:
                        print(f"Multimodal fusion error: {e}")
                        # Fallback to averaging individual results
                        avg_probs = (facial_emotion_result['probabilities'] + audio_emotion_result['probabilities']) / 2
                        combined_emotion_result = {
                            'emotion': ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'][np.argmax(avg_probs)],
                            'confidence': avg_probs.max(),
                            'probabilities': avg_probs
                        }
                else:
                    # Use the available result or neutral fallback
                    available_result = facial_emotion_result if facial_input_tensor is not None else audio_emotion_result
                    combined_emotion_result = available_result
                
                print(f"[DEBUG] Combined Emotion Result: {combined_emotion_result}") # Added debug print
                
                # Determine dominant emotion
                dominant_emotion, dominant_confidence = self._determine_dominant_emotion(
                    facial_emotion_result, audio_emotion_result, combined_emotion_result
                )
                
                # Update environment controller
                self.env_controller.update_emotion(dominant_emotion, dominant_confidence)
                
                # Generate dialogue using DeepSeek model
                dialogue = ""
                current_time = time.time()
                print(f"[DEBUG] Before _should_generate_dialogue call. Dominant Emotion: {dominant_emotion}, Confidence: {dominant_confidence:.2f}")
                if self._should_generate_dialogue(dominant_emotion, dominant_confidence, current_time):
                    dialogue = self.env_controller.generate_dialogue(dominant_emotion) # Placeholder for actual LLM call
                    self.last_dialogue_time = current_time
                    self.last_dialogue_emotion = dominant_emotion
                    print(f"[DEBUG] Generated dialogue: {dialogue}") # Debug print
                
                # Update frame count (fix double increment bug)
                self.frame_count += 1
                
                # Put processed data into the queue for Streamlit UI
                try:
                    data_queue.put_nowait({
                        'type': 'update',
                        'facial_emotion_name': facial_emotion_result['emotion'],
                        'facial_probs': facial_emotion_result['probabilities'],
                        'audio_emotion_name': audio_emotion_result['emotion'],
                        'audio_probs': audio_emotion_result['probabilities'],
                        'combined_emotion_name': combined_emotion_result['emotion'],
                        'combined_probs': combined_emotion_result['probabilities'],
                        'dominant_emotion': dominant_emotion,
                        'dominant_confidence': dominant_confidence,
                        'dialogue': dialogue,
                        'frame_count': self.frame_count
                    })
                    print(f"[DEBUG] Pushed update to queue: Emotion={dominant_emotion}, Frame={self.frame_count}, Dialogue='{dialogue[:30] if dialogue else 'None'}...'") # Debug print
                except queue.Full:
                    # Queue is full, skip this update to prevent blocking
                    print("[WARNING] Queue is full, skipping update")
                    pass
                
                # Update accuracy metrics (removed duplicate frame count increment)
                if self.frame_count % self.accuracy_update_interval == 0:
                    self._calculate_accuracy_metrics(
                        facial_emotion_result['probabilities'],
                        audio_emotion_result['probabilities'],
                        combined_emotion_result['probabilities']
                    )
                    data_queue.put({
                        'type': 'update',
                        'frame_count': self.frame_count,
                        'avg_facial_entropy': np.mean(self.facial_accuracies) if self.facial_accuracies else 0,
                        'avg_audio_entropy': np.mean(self.audio_accuracies) if self.audio_accuracies else 0,
                        'avg_combined_entropy': np.mean(self.combined_accuracies) if self.combined_accuracies else 0
                    })
                    print("[DEBUG] Updated accuracy metrics in queue")
                
                # Introduce a small delay to control processing rate
                time.sleep(0.05) # Adjust as needed for performance
                
            except Exception as e:
                data_queue.put({'type': 'error', 'message': f"Real-time processing error: {e}"})
                print(f"Critical real-time loop error: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for debugging
                stop_event.set() # Stop the loop on critical errors
        
        # Cleanup
        # self._print_accuracy_summary() # This will be removed in the next step
        data_queue.put({'type': 'info', 'message': "Real-time adaptation stopped."})
    
    def _should_generate_dialogue(self, emotion, confidence, current_time):
        """Determine if dialogue should be generated based on timing and emotion changes."""
        # Always allow first dialogue
        if self.last_dialogue_time == 0:
            print(f"[DEBUG] _should_generate_dialogue: First dialogue, returning True.")
            return True
        
        # Check cooldown period
        if current_time - self.last_dialogue_time < self.dialogue_cooldown:
            print(f"[DEBUG] _should_generate_dialogue: Cooldown active. Returning False. Time elapsed: {current_time - self.last_dialogue_time:.2f}s")
            return False
        
        # Allow dialogue if emotion changed or confidence is high
        emotion_changed = emotion != self.last_dialogue_emotion
        high_confidence = confidence > self.confidence_threshold # Changed to use self.confidence_threshold
        
        print(f"[DEBUG] _should_generate_dialogue: Emotion changed: {emotion_changed} (current: {emotion}, last: {self.last_dialogue_emotion}), High confidence ({confidence:.2f} > {self.confidence_threshold}): {high_confidence}. Returning {emotion_changed or high_confidence}.")
        
        return emotion_changed or high_confidence

    def release_resources(self):
        """Release camera and audio resources safely."""
        try:
            if hasattr(self, 'video_capture') and self.video_capture:
                self.video_capture.release()
                self.video_capture = None
        except Exception as e:
            logging.warning(f"Error releasing video capture: {e}")
        
        try:
            if hasattr(self, 'audio_stream') and self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
        except Exception as e:
            logging.warning(f"Error releasing audio stream: {e}")
        
        try:
            if hasattr(self, 'pyaudio_instance') and self.pyaudio_instance:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
        except Exception as e:
            logging.warning(f"Error terminating PyAudio: {e}")
        
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logging.warning(f"Error destroying OpenCV windows: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with guaranteed cleanup."""
        self.release_resources()

    def _apply_temporal_smoothing(self, current_probs):
        """Apply temporal smoothing to emotion probabilities."""
        self.emotion_history.append(current_probs)
        smoothed_probs = np.mean(list(self.emotion_history), axis=0)
        dominant_emotion_idx = np.argmax(smoothed_probs)
        confidence = smoothed_probs[dominant_emotion_idx]
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        dominant_emotion = emotions[dominant_emotion_idx]
        return {'emotion': dominant_emotion, 'confidence': confidence, 'probabilities': smoothed_probs}

    def _determine_dominant_emotion(self, facial_result, audio_result, combined_result):
        """Determine the dominant emotion from multiple sources, prioritizing combined.
        Assumes each result has 'emotion', 'confidence', and 'probabilities'."""
        
        # Start with combined result if its confidence is high enough
        if combined_result['confidence'] >= self.confidence_threshold:
            return combined_result['emotion'], combined_result['confidence']
        
        # If not, consider the individual strongest emotion
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        all_probs = np.array([
            facial_result['probabilities'],
            audio_result['probabilities'],
        ])
        
        # Average probabilities (or implement more sophisticated fusion)
        avg_probs = np.mean(all_probs, axis=0)
        
        dominant_idx = np.argmax(avg_probs)
        dominant_emotion = emotions[dominant_idx]
        dominant_confidence = avg_probs[dominant_idx]
        
        # Apply temporal smoothing to the dominant emotion for stability
        smoothed_result = self._apply_temporal_smoothing(avg_probs)

        return smoothed_result['emotion'], smoothed_result['confidence']

    def _should_update_emotion(self, new_emotion, new_confidence):
        """Determine if the environment emotion should be updated."""
        if self.last_emotion is None:
            return True
        
        # Only update if emotion changes and confidence is above threshold, or if it's a significant change
        return (new_emotion != self.last_emotion and new_confidence >= self.confidence_threshold) or \
               (new_confidence > self.last_confidence + 0.1 and new_emotion == self.last_emotion)

    def _calculate_accuracy_metrics(self, facial_probs, audio_probs, combined_probs):
        """Calculate and store accuracy metrics (dummy for real-time, needs true labels for real accuracy)."""
        # In a real-time system without ground truth, this would be complex.
        # For demonstration, we calculate entropy as a proxy for uncertainty.
        # Removed duplicate frame_count increment - it's already handled in the main loop
        
        # Example of how entropy could be calculated (for diversity/uncertainty)
        def calculate_entropy(probs):
            # Avoid log(0)
            epsilon = 1e-10
            return -np.sum(probs * np.log2(probs + epsilon))

        self.facial_accuracies.append(calculate_entropy(facial_probs)) # Example: use entropy as a proxy
        self.audio_accuracies.append(calculate_entropy(audio_probs))
        self.combined_accuracies.append(calculate_entropy(combined_probs)) 