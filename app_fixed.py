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
import uuid
from streamlit_autorefresh import st_autorefresh

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
    QUEUE_MAX_SIZE = 50
    QUEUE_PROCESS_LIMIT = 5
    
    # Model names
    OLLAMA_MODEL = "deepseek-r1:latest"

# Global thread management
class ThreadManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.threads = {}
                    cls._instance.stop_events = {}
        return cls._instance
    
    def start_thread(self, session_id, target, args):
        if session_id in self.threads and self.threads[session_id].is_alive():
            return False  # Thread already running
        
        stop_event = threading.Event()
        thread = threading.Thread(target=target, args=(*args, stop_event))
        thread.daemon = True
        thread.start()
        
        self.threads[session_id] = thread
        self.stop_events[session_id] = stop_event
        return True
    
    def stop_thread(self, session_id):
        if session_id in self.stop_events:
            self.stop_events[session_id].set()
        if session_id in self.threads:
            del self.threads[session_id]
        if session_id in self.stop_events:
            del self.stop_events[session_id]
    
    def is_thread_running(self, session_id):
        return (session_id in self.threads and 
                self.threads[session_id].is_alive())

# Initialize global thread manager
thread_manager = ThreadManager()

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

# Load models once
facial_model = load_facial_model()
audio_model = load_audio_model()
fusion_model = load_fusion_model()

# --- Main Streamlit App --- #
def run_streamlit_app():
    # Set page config first (only once)
    st.set_page_config(page_title="VR Emotion Adaptation", layout="centered")
    
    # Generate unique session ID
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    session_id = st.session_state.session_id
    
    st.title("VR Emotion Adaptation System")
    st.sidebar.title("🎮 Controls")
    
    # Initialize session state variables
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.last_emotion = 'N/A'
        st.session_state.last_confidence = 'N/A'
        st.session_state.last_dialogue = 'Waiting for emotion detection...'
        st.session_state.frame_count = 0
        st.session_state.webcam_status = "Unknown"
        st.session_state.audio_status = "Unknown"
        st.session_state.data_queue = queue.Queue(maxsize=AppConfig.QUEUE_MAX_SIZE)
    
    # Auto-refresh only when thread is running (reduced frequency)
    if thread_manager.is_thread_running(session_id):
        st_autorefresh(interval=AppConfig.AUTOREFRESH_INTERVAL_MS, key="realtime_refresh")
    
    # Check Ollama connection (cached to avoid repeated calls)
    @st.cache_data(ttl=30)  # Cache for 30 seconds
    def check_ollama_status():
        try:
            return check_ollama_and_model("deepseek-r1:latest")
        except Exception as e:
            return False, f"Error checking Ollama: {str(e)}"
    
    ollama_ok, ollama_msg = check_ollama_status()
    
    # Initialize VR app (cached)
    @st.cache_resource
    def get_vr_app():
        return VREmotionAdaptation(facial_model, audio_model, fusion_model)
    
    try:
        app = get_vr_app()
    except Exception as e:
        st.error(f"Failed to initialize VR application: {e}")
        return
    
    # --- Status Display ---
    st.sidebar.header("System Status")
    st.sidebar.markdown(f"**Ollama Model:** {'✅ Available' if ollama_ok else '❌ Not Available'}")
    st.sidebar.markdown(f"**Processing:** {'✅ Running' if thread_manager.is_thread_running(session_id) else '❌ Stopped'}")
    st.sidebar.markdown(f"**Webcam:** {st.session_state.webcam_status}")
    st.sidebar.markdown(f"**Audio:** {st.session_state.audio_status}")
    
    # --- Main Display ---
    col1, col2 = st.columns(2)
    
    with col1:
        emotion_container = st.container()
        with emotion_container:
            st.metric("Current Emotion", st.session_state.last_emotion)
    
    with col2:
        confidence_container = st.container()
        with confidence_container:
            if isinstance(st.session_state.last_confidence, (int, float)):
                st.metric("Confidence", f"{st.session_state.last_confidence:.2f}")
            else:
                st.metric("Confidence", str(st.session_state.last_confidence))
    
    # Dialogue display
    dialogue_container = st.container()
    with dialogue_container:
        st.subheader("NPC Dialogue")
        if st.session_state.last_dialogue != 'Waiting for emotion detection...':
            st.success(f"🤖 {st.session_state.last_dialogue}")
        else:
            st.info(f"💬 {st.session_state.last_dialogue}")
    
    # Progress indicator
    if thread_manager.is_thread_running(session_id):
        progress_container = st.container()
        with progress_container:
            progress_value = min((st.session_state.frame_count % 100) / 100.0, 1.0)
            st.progress(progress_value)
            st.caption(f"Frames processed: {st.session_state.frame_count}")
    
    # --- Controls ---
    st.sidebar.markdown("### Controls")
    
    # Start button
    if not thread_manager.is_thread_running(session_id):
        if st.sidebar.button("🚀 Start VR Adaptation", disabled=not ollama_ok):
            with st.spinner("Initializing capture devices..."):
                try:
                    success, status_msg = app.setup_realtime_capture()
                    
                    # Update device status
                    st.session_state.webcam_status = "✅ Available" if app.video_capture else "❌ Not Available"
                    st.session_state.audio_status = "✅ Available" if app.audio_stream else "❌ Not Available"
                    
                    if app.video_capture or app.audio_stream:
                        # Start processing thread
                        if thread_manager.start_thread(
                            session_id, 
                            app.run_realtime_adaptation, 
                            (st.session_state.data_queue,)
                        ):
                            st.sidebar.success("✅ Started successfully!")
                            st.rerun()
                        else:
                            st.sidebar.error("❌ Failed to start thread")
                    else:
                        st.sidebar.error("❌ No capture devices available")
                        
                except Exception as e:
                    st.sidebar.error(f"❌ Setup failed: {e}")
    else:
        if st.sidebar.button("⏹️ Stop VR Adaptation"):
            thread_manager.stop_thread(session_id)
            app.release_resources()
            st.sidebar.success("✅ Stopped successfully!")
            st.rerun()
    
    # --- Process Queue Data ---
    if thread_manager.is_thread_running(session_id):
        data_processed = 0
        max_process = 5  # Limit processing per refresh to prevent blocking
        
        while not st.session_state.data_queue.empty() and data_processed < max_process:
            try:
                data = st.session_state.data_queue.get_nowait()
                data_processed += 1
                
                if data.get('type') == 'update':
                    # Update emotion
                    if 'combined_emotion_name' in data or 'facial_emotion_name' in data or 'audio_emotion_name' in data:
                        new_emotion = (data.get('combined_emotion_name') or 
                                     data.get('facial_emotion_name') or 
                                     data.get('audio_emotion_name', 'N/A'))
                        if new_emotion != st.session_state.last_emotion:
                            st.session_state.last_emotion = new_emotion
                    
                    # Update confidence
                    if 'dominant_confidence' in data:
                        st.session_state.last_confidence = data.get('dominant_confidence', 'N/A')
                    
                    # Update dialogue
                    if 'dialogue' in data and data.get('dialogue'):
                        st.session_state.last_dialogue = data.get('dialogue', 'N/A')
                    
                    # Update frame count
                    if 'frame_count' in data:
                        st.session_state.frame_count = data.get('frame_count', 0)
                
                elif data.get('type') == 'error':
                    st.sidebar.error(f"❌ {data.get('message', 'Unknown error')}")
                elif data.get('type') == 'info':
                    st.sidebar.info(f"ℹ️ {data.get('message', '')}")
                    
            except queue.Empty:
                break
            except Exception as e:
                print(f"[ERROR] Queue processing error: {e}")
                break
    
    # --- Status Messages ---
    if not thread_manager.is_thread_running(session_id):
        if not ollama_ok:
            st.warning("⚠️ Ollama is not available. Please start Ollama and ensure the DeepSeek model is installed.")
            st.info("💡 Run: `ollama pull deepseek-r1:latest` to install the required model.")
        else:
            st.info("🎯 Click 'Start VR Adaptation' to begin real-time emotion detection.")
    else:
        st.success("🔄 Real-time emotion detection is active!")
    
    # --- Debug Section ---
    with st.expander("🔧 Debug Information", expanded=False):
        st.json({
            "Session ID": session_id,
            "Thread Running": thread_manager.is_thread_running(session_id),
            "Ollama Status": ollama_ok,
            "Queue Size": st.session_state.data_queue.qsize(),
            "Frame Count": st.session_state.frame_count,
            "Last Emotion": st.session_state.last_emotion,
            "Last Confidence": st.session_state.last_confidence
        })

if __name__ == "__main__":
    run_streamlit_app()