import streamlit as st
import os
import torch
import numpy as np
import cv2 as cv
import librosa
import pyaudio
import time
import threading
import queue
import logging
from streamlit_autorefresh import st_autorefresh

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Construct the path to haarcascades directly
HAARCASCADE_PATH = os.path.join(os.path.dirname(cv.__file__), 'data', 'haarcascades')

from data.data_loader import FacialExpressionDataset, AudioEmotionDataset
from models.emotion_models import FacialEmotionCNN, AudioEmotionLSTM, MultiModalEmotionFusion, EmotionClassifier
from vr_adaptation.environment_controller import EnvironmentController, EmotionType

from utils import check_ollama_and_model
from vr_emotion_adaptation import VREmotionAdaptation

# --- Configuration --- #
class AppConfig:
    """Centralized configuration for the VR Emotion Adaptation app."""
    # Data paths
    FACIAL_DATA_PATH = "archive (1)/test"
    AUDIO_DATA_PATH = "archive/audio_speech_actors_01-24"
    
    # Model weights paths
    FACIAL_MODEL_WEIGHTS = "models/facial_emotion_cnn.pth"
    AUDIO_MODEL_WEIGHTS = "models/audio_emotion_lstm.pth"
    MULTIMODAL_MODEL_WEIGHTS = "models/multimodal_emotion_fusion.pth"
    
    # Processing parameters
    BATCH_SIZE = 1
    SAMPLE_RATE = 22050
    AUDIO_CHUNK_SIZE = 4096
    
    # UI refresh rates
    AUTOREFRESH_INTERVAL_MS = 500
    QUEUE_MAX_SIZE = 10
    
    # Model names
    OLLAMA_MODEL = "deepseek-r1:latest"

# --- Session State Management --- #
def _initialize_session_state():
    """Initialize all session state variables with default values."""
    if 'app_initialized' not in st.session_state:
        # Application state
        st.session_state.app_initialized = False
        st.session_state.thread_started = False
        
        # UI state
        st.session_state.last_emotion = 'N/A'
        st.session_state.last_confidence = 'N/A'
        st.session_state.last_dialogue = 'Waiting for emotion detection...'
        
        # Hardware status
        st.session_state.webcam_status = "Unknown (start to check)"
        st.session_state.audio_status = "Unknown (start to check)"
        
        # Processing state
        st.session_state.frame_count = 0
        st.session_state.data_queue = queue.Queue(maxsize=AppConfig.QUEUE_MAX_SIZE)
        st.session_state.stop_event = threading.Event()
        
        logger.debug("Session state initialized")

def _ensure_logs_directory():
    """Ensure logs directory exists before logging."""
    if not os.path.exists('logs'):
        os.makedirs('logs')
        logger.info("Created logs directory")

def _render_main_display():
    """Render the main emotion detection display components."""
    st.subheader("Real-Time Emotion Detection")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Current Emotion")
        emotion_display = st.empty()
        
    with col2:
        st.markdown("### Confidence Level")
        confidence_display = st.empty()
    
    st.markdown("### NPC Dialogue Response")
    dialogue_display = st.empty()
    
    # Initialize displays with current values
    current_emotion = st.session_state.get('last_emotion', 'N/A')
    current_confidence = st.session_state.get('last_confidence', 'N/A')
    current_dialogue = st.session_state.get('last_dialogue', 'Waiting for emotion detection...')
    
    emotion_display.metric("Detected Emotion", current_emotion)
    if isinstance(current_confidence, (int, float)):
        confidence_display.metric("Confidence", f"{current_confidence:.2f}")
    else:
        confidence_display.metric("Confidence", str(current_confidence))
    dialogue_display.info(f"💬 {current_dialogue}")
    
    return emotion_display, confidence_display, dialogue_display

# --- UI Update Function --- #
def update_ui(data, placeholders):
    emotion_placeholder, dialogue_placeholder, status_placeholder = placeholders
    if data.get('type') == 'update':
        # Show only the most relevant emotion and dialogue
        emotion = data.get('combined_emotion_name') or data.get('facial_emotion_name') or data.get('audio_emotion_name', 'N/A')
        confidence = data.get('dominant_confidence', 0)
        emotion_placeholder.markdown(f"**Detected Emotion:** {emotion} (Confidence: {confidence:.2f})")
        dialogue = data.get('dialogue', '').strip() if data.get('dialogue') else ''
        if dialogue:
            dialogue_placeholder.success(f"🤖 **NPC Dialogue:** {dialogue}")
        else:
            dialogue_placeholder.info("Waiting for dialogue generation or Ollama connection...")
    elif data.get('type') == 'error':
        status_placeholder.error(data.get('message', 'Unknown error'))
    elif data.get('type') == 'info':
        status_placeholder.info(data.get('message', ''))

@st.cache_resource
def load_facial_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FacialEmotionCNN()
    model.load_state_dict(torch.load("models/facial_emotion_cnn.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_audio_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AudioEmotionLSTM()
    model.load_state_dict(torch.load("models/audio_emotion_lstm.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_fusion_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiModalEmotionFusion()
    model.load_state_dict(torch.load("models/multimodal_emotion_fusion.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# Models will be loaded on-demand in the VREmotionAdaptation class

# --- Main Streamlit App --- #
def run_streamlit_app():
    # Set page config first - only once
    st.set_page_config(page_title="VR Emotion Adaptation", layout="centered")
    st.title("VR Emotion Adaptation System")
    
    # Ensure logs directory exists
    _ensure_logs_directory()
    
    # Initialize session state variables first to prevent resets
    _initialize_session_state()

    # Only auto-refresh when thread is running to prevent unnecessary reruns
    if st.session_state.get('thread_started', False):
        st_autorefresh(interval=AppConfig.AUTOREFRESH_INTERVAL_MS, key="app_autorefresh")  # Faster refresh only when needed

    # Check Ollama connection and model (with error handling) - cache the result
    if 'ollama_status' not in st.session_state:
        try:
            ollama_ok, ollama_msg = check_ollama_and_model(AppConfig.OLLAMA_MODEL)
            st.session_state.ollama_status = (ollama_ok, ollama_msg)
        except Exception as e:
            ollama_ok = False
            ollama_msg = f"Error checking Ollama: {str(e)}"
            st.session_state.ollama_status = (ollama_ok, ollama_msg)
            logger.error(f"Ollama check failed: {e}")
    else:
        ollama_ok, ollama_msg = st.session_state.ollama_status
    
    # Initialize the main application class only once per session
    if 'vr_app' not in st.session_state:
        try:
            st.session_state.vr_app = VREmotionAdaptation()
            st.session_state.app_initialized = True
            logger.debug("VR app initialized")
        except Exception as e:
            st.error(f"Failed to initialize VR application: {e}")
            st.sidebar.error("App Init Failed")
            return
    
    app = st.session_state.vr_app

    # Placeholders for real-time data
    emotion_placeholder = st.empty()
    dialogue_placeholder = st.empty()
    status_placeholder = st.sidebar.empty()

    # Session state already initialized in _initialize_session_state()
        
    # Debug logging for button logic conditions
    logger.debug(f"current st.session_state.thread_started: {st.session_state.thread_started}")
    logger.debug(f"current ollama_ok: {ollama_ok}")
        
    # --- Always-visible status/info section ---
    st.sidebar.header("System Status")
    st.sidebar.markdown(f"**Ollama Model Status:** {'✅ Available' if ollama_ok else '❌ Not Available'}")
    st.sidebar.markdown(f"**Thread Running:** {'✅ Yes' if st.session_state.get('thread_started', False) else '❌ No'}")
    
    # Hardware status already initialized in _initialize_session_state()
        
    st.sidebar.markdown(f"**Webcam Status:** {st.session_state['webcam_status']}")
    st.sidebar.markdown(f"**Audio Status:** {st.session_state['audio_status']}")

    # --- Main Feature Display ---
    emotion_display, confidence_display, dialogue_display = _render_main_display()

    # --- Controls ---
    st.sidebar.markdown("### Actions")
    
    # Always show the start button, but disable if Ollama is not available
    start_disabled = not ollama_ok
    start_button_text = "Start VR Adaptation" if ollama_ok else "Start VR Adaptation (Ollama Required)"
    
    if st.sidebar.button(start_button_text, disabled=start_disabled, key="start_btn"):
        logger.debug("Start VR Adaptation button clicked.")
        if not st.session_state.thread_started:
            st.session_state.stop_event.clear()
            try:
                success, status_msg = app.setup_realtime_capture()
                logger.debug(f"Real-time capture setup initiated. Success: {success}, Status: {status_msg}")
                
                # Update the webcam and audio status in the sidebar
                webcam_status = "✅ Available" if app.video_capture else "❌ Not Available"
                audio_status = "✅ Available" if app.audio_stream else "❌ Not Available"
                
                st.session_state['webcam_status'] = webcam_status
                st.session_state['audio_status'] = audio_status
                
                # Allow starting even if only one input is available
                if app.video_capture or app.audio_stream:
                    try:
                        # First make sure we have the correct thread parameters
                        if not isinstance(st.session_state.stop_event, threading.Event):
                            logger.warning("Recreating stop_event as it's not a proper Event")
                            st.session_state.stop_event = threading.Event()
                            
                        # Create and start the processing thread
                        processing_thread = threading.Thread(
                            target=app.run_realtime_adaptation,
                            args=(st.session_state.stop_event, st.session_state.data_queue)
                        )
                        processing_thread.daemon = True
                        processing_thread.start()
                        
                        # Update the session state
                        st.session_state.thread_started = True
                        status_placeholder.success("Real-time adaptation started!")
                        logger.info("Processing thread started successfully")
                        
                        # Force a rerun to update the UI immediately
                        st.rerun()
                    except Exception as e:
                        logger.error(f"Failed to start processing thread: {e}")
                        st.error(f"Failed to start processing: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    st.error("Neither webcam nor audio is available. Please check your devices.")
                    
            except Exception as e:
                st.error(f"Error during real-time capture setup: {e}")
                logger.error(f"Real-time capture setup failed: {e}")
                st.session_state.thread_started = False
    
    if st.sidebar.button("Stop VR Adaptation", key="stop_btn"):
        if st.session_state.thread_started:
            st.session_state.stop_event.set()
            app.release_resources()
            st.session_state.thread_started = False
            status_placeholder.warning("Real-time adaptation stopped.")
    
    # Add a test button to verify sidebar is working
    if st.sidebar.button("Test Button", key="test_btn"):
        st.sidebar.success("Sidebar is working!")

    # --- Real-time UI update from queue ---
    if not st.session_state.thread_started:
        st.info("Click 'Start VR Adaptation' to begin.")
    else:
        # Process all available data in the queue
        data_updated = False
        while not st.session_state.data_queue.empty():
            data = st.session_state.data_queue.get()
            data_updated = True
            
            # Update session state for always-on display
            if data.get('type') == 'update':
                if 'combined_emotion_name' in data or 'facial_emotion_name' in data or 'audio_emotion_name' in data:
                    new_emotion = data.get('combined_emotion_name') or data.get('facial_emotion_name') or data.get('audio_emotion_name', 'N/A')
                    st.session_state['last_emotion'] = new_emotion
                    emotion_display.metric("Detected Emotion", new_emotion)
                    
                if 'dominant_confidence' in data:
                    new_confidence = data.get('dominant_confidence', 'N/A')
                    st.session_state['last_confidence'] = new_confidence
                    if isinstance(new_confidence, (int, float)):
                        confidence_display.metric("Confidence", f"{new_confidence:.2f}")
                    else:
                        confidence_display.metric("Confidence", str(new_confidence))
                        
                if 'dialogue' in data and data.get('dialogue'):
                    new_dialogue = data.get('dialogue', 'N/A')
                    st.session_state['last_dialogue'] = new_dialogue
                    dialogue_display.success(f"🤖 **NPC Response:** {new_dialogue}")
                
                # Update frame count
                if 'frame_count' in data:
                    st.session_state['frame_count'] = data.get('frame_count', 0)
                    
            elif data.get('type') == 'error':
                status_placeholder.error(data.get('message', 'Unknown error'))
            elif data.get('type') == 'info':
                status_placeholder.info(data.get('message', ''))
        
        # Show status if no data is being processed
        if not data_updated and st.session_state.data_queue.empty():
            st.info("🔄 Processing real-time data... Make sure your webcam and microphone are working.")
            
    # Add a progress indicator for real-time processing
    if st.session_state.thread_started:
        progress_bar = st.progress(0)
        frame_count = st.session_state.get('frame_count', 0)
        progress_value = min((frame_count % 100) / 100.0, 1.0)
        progress_bar.progress(progress_value)
        st.caption(f"Frames processed: {frame_count}")

    # --- Debug/Logs Section ---
    with st.expander("Debug / Logs", expanded=False):
        st.write(f"[DEBUG] Ollama models found: {['deepseek-r1:latest', 'qwen2.5:latest']}")
        st.write(f"[DEBUG] current st.session_state.thread_started: {st.session_state.thread_started}")
        st.write(f"[DEBUG] current ollama_ok: {ollama_ok}")

if __name__ == "__main__":
    run_streamlit_app() 