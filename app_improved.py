"""
Improved VR Emotion Adaptation System
Addresses architectural issues and improves maintainability
"""

import streamlit as st
import logging
import threading
import queue
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from streamlit_autorefresh import st_autorefresh

from vr_emotion_adaptation_improved import VREmotionAdaptation
from utils import check_ollama_and_model


@dataclass
class UIConfig:
    """UI-specific configuration"""
    autorefresh_interval_ms: int = 500
    queue_max_size: int = 50
    queue_process_limit: int = 5
    page_title: str = "VR Emotion Adaptation"


@dataclass
class AppState:
    """Centralized application state"""
    current_emotion: str = 'N/A'
    confidence: float = 0.0
    dialogue: str = 'Waiting for emotion detection...'
    frame_count: int = 0
    webcam_status: str = "Unknown"
    audio_status: str = "Unknown"
    processing_active: bool = False
    ollama_available: bool = False


class UIManager:
    """Manages UI components and updates"""
    
    def __init__(self, config: UIConfig):
        self.config = config
        self._setup_page()
    
    def _setup_page(self):
        """Setup page configuration"""
        st.set_page_config(
            page_title=self.config.page_title,
            layout="centered"
        )
        st.title(self.config.page_title)
    
    def render_status_sidebar(self, state: AppState):
        """Render system status in sidebar"""
        st.sidebar.header("System Status")
        st.sidebar.markdown(f"**Ollama Model:** {'✅ Available' if state.ollama_available else '❌ Not Available'}")
        st.sidebar.markdown(f"**Processing:** {'✅ Running' if state.processing_active else '❌ Stopped'}")
        st.sidebar.markdown(f"**Webcam:** {state.webcam_status}")
        st.sidebar.markdown(f"**Audio:** {state.audio_status}")
    
    def render_emotion_display(self, state: AppState):
        """Render main emotion detection display"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current Emotion", state.current_emotion)
        
        with col2:
            if isinstance(state.confidence, (int, float)):
                st.metric("Confidence", f"{state.confidence:.2f}")
            else:
                st.metric("Confidence", str(state.confidence))
        
        st.subheader("NPC Dialogue")
        if state.dialogue != 'Waiting for emotion detection...':
            st.success(f"🤖 {state.dialogue}")
        else:
            st.info(f"💬 {state.dialogue}")
    
    def render_progress_indicator(self, state: AppState):
        """Render processing progress"""
        if state.processing_active:
            progress_value = min((state.frame_count % 100) / 100.0, 1.0)
            st.progress(progress_value)
            st.caption(f"Frames processed: {state.frame_count}")
    
    def render_controls(self, state: AppState) -> Dict[str, bool]:
        """Render control buttons and return their states"""
        st.sidebar.markdown("### Controls")
        
        controls = {}
        
        if not state.processing_active:
            controls['start'] = st.sidebar.button(
                "🚀 Start VR Adaptation",
                disabled=not state.ollama_available
            )
        else:
            controls['stop'] = st.sidebar.button("⏹️ Stop VR Adaptation")
        
        return controls


class ProcessingManager:
    """Manages processing thread lifecycle"""
    
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
    
    def start_processing(self, vr_app: VREmotionAdaptation, data_queue: queue.Queue) -> bool:
        """Start processing thread"""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._safe_processing_wrapper,
                args=(vr_app, data_queue)
            )
            self._thread.daemon = True
            self._thread.start()
            return True
    
    def stop_processing(self):
        """Stop processing thread"""
        with self._lock:
            if self._thread:
                self._stop_event.set()
                self._thread.join(timeout=5.0)
    
    def is_active(self) -> bool:
        """Check if processing is active"""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()
    
    def _safe_processing_wrapper(self, vr_app: VREmotionAdaptation, data_queue: queue.Queue):
        """Wrapper for safe thread execution"""
        try:
            vr_app.run_realtime_adaptation(self._stop_event, data_queue)
        except Exception as e:
            logging.error(f"Processing thread failed: {e}")
            try:
                data_queue.put_nowait({'type': 'error', 'message': str(e)})
            except queue.Full:
                pass


class StateManager:
    """Manages application state updates"""
    
    def __init__(self):
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state if not already done"""
        if 'app_state' not in st.session_state:
            st.session_state.app_state = AppState()
            st.session_state.data_queue = queue.Queue(maxsize=UIConfig().queue_max_size)
    
    def get_state(self) -> AppState:
        """Get current application state"""
        return st.session_state.app_state
    
    def update_state(self, **kwargs):
        """Update application state"""
        for key, value in kwargs.items():
            if hasattr(st.session_state.app_state, key):
                setattr(st.session_state.app_state, key, value)
    
    def process_queue_updates(self, data_queue: queue.Queue, max_updates: int = 5):
        """Process updates from the data queue"""
        updates_processed = 0
        
        while not data_queue.empty() and updates_processed < max_updates:
            try:
                data = data_queue.get_nowait()
                updates_processed += 1
                
                if data.get('type') == 'update':
                    self._process_emotion_update(data)
                elif data.get('type') == 'error':
                    st.sidebar.error(f"❌ {data.get('message', 'Unknown error')}")
                elif data.get('type') == 'info':
                    st.sidebar.info(f"ℹ️ {data.get('message', '')}")
                    
            except queue.Empty:
                break
            except Exception as e:
                logging.error(f"Queue processing error: {e}")
                break
    
    def _process_emotion_update(self, data: Dict[str, Any]):
        """Process emotion update from queue"""
        # Update emotion
        emotion = (data.get('combined_emotion_name') or 
                  data.get('facial_emotion_name') or 
                  data.get('audio_emotion_name', 'N/A'))
        
        confidence = data.get('dominant_confidence', 0.0)
        dialogue = data.get('dialogue', '')
        frame_count = data.get('frame_count', 0)
        
        self.update_state(
            current_emotion=emotion,
            confidence=confidence,
            frame_count=frame_count
        )
        
        if dialogue:
            self.update_state(dialogue=dialogue)


class VREmotionApp:
    """Main application class with improved architecture"""
    
    def __init__(self):
        self.config = UIConfig()
        self.ui_manager = UIManager(self.config)
        self.state_manager = StateManager()
        self.processing_manager = ProcessingManager()
        self.vr_app: Optional[VREmotionAdaptation] = None
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup application logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @st.cache_resource
    def _get_vr_app(_self) -> VREmotionAdaptation:
        """Get VR application instance (cached)"""
        return VREmotionAdaptation()
    
    @st.cache_data(ttl=30)
    def _check_ollama_status(_self) -> Tuple[bool, str]:
        """Check Ollama status (cached)"""
        try:
            return check_ollama_and_model("deepseek-r1:latest")
        except Exception as e:
            return False, f"Error checking Ollama: {str(e)}"
    
    def run(self):
        """Main application entry point"""
        # Initialize VR app
        try:
            self.vr_app = self._get_vr_app()
        except Exception as e:
            st.error(f"Failed to initialize VR application: {e}")
            return
        
        # Check Ollama status
        ollama_ok, ollama_msg = self._check_ollama_status()
        self.state_manager.update_state(ollama_available=ollama_ok)
        
        # Update processing status
        processing_active = self.processing_manager.is_active()
        self.state_manager.update_state(processing_active=processing_active)
        
        # Get current state
        state = self.state_manager.get_state()
        
        # Auto-refresh when processing
        if state.processing_active:
            st_autorefresh(interval=self.config.autorefresh_interval_ms, key="app_refresh")
        
        # Render UI
        self.ui_manager.render_status_sidebar(state)
        self.ui_manager.render_emotion_display(state)
        self.ui_manager.render_progress_indicator(state)
        
        # Handle controls
        controls = self.ui_manager.render_controls(state)
        self._handle_controls(controls)
        
        # Process queue updates
        if state.processing_active:
            self.state_manager.process_queue_updates(
                st.session_state.data_queue,
                self.config.queue_process_limit
            )
        
        # Show status messages
        self._show_status_messages(state)
    
    def _handle_controls(self, controls: Dict[str, bool]):
        """Handle control button presses"""
        if controls.get('start'):
            self._start_processing()
        elif controls.get('stop'):
            self._stop_processing()
    
    def _start_processing(self):
        """Start emotion processing"""
        with st.spinner("Initializing capture devices..."):
            try:
                success, status_msg = self.vr_app.setup_realtime_capture()
                
                # Update device status
                webcam_status = "✅ Available" if self.vr_app.resource_manager.video_available else "❌ Not Available"
                audio_status = "✅ Available" if self.vr_app.resource_manager.audio_available else "❌ Not Available"
                
                self.state_manager.update_state(
                    webcam_status=webcam_status,
                    audio_status=audio_status
                )
                
                if success:
                    if self.processing_manager.start_processing(self.vr_app, st.session_state.data_queue):
                        st.sidebar.success("✅ Started successfully!")
                        st.rerun()
                    else:
                        st.sidebar.error("❌ Failed to start processing thread")
                else:
                    st.sidebar.error("❌ No capture devices available")
                    
            except Exception as e:
                st.sidebar.error(f"❌ Setup failed: {e}")
    
    def _stop_processing(self):
        """Stop emotion processing"""
        self.processing_manager.stop_processing()
        self.vr_app.release_resources()
        st.sidebar.success("✅ Stopped successfully!")
        st.rerun()
    
    def _show_status_messages(self, state: AppState):
        """Show appropriate status messages"""
        if not state.processing_active:
            if not state.ollama_available:
                st.warning("⚠️ Ollama is not available. Please start Ollama and ensure the DeepSeek model is installed.")
                st.info("💡 Run: `ollama pull deepseek-r1:latest` to install the required model.")
            else:
                st.info("🎯 Click 'Start VR Adaptation' to begin real-time emotion detection.")
        else:
            st.success("🔄 Real-time emotion detection is active!")


def main():
    """Application entry point"""
    app = VREmotionApp()
    app.run()


if __name__ == "__main__":
    main()