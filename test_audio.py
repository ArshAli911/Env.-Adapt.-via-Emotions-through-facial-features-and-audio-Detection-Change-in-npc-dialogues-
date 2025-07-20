import pyaudio
import numpy as np
import time

# Audio parameters
SAMPLE_RATE = 22050
AUDIO_CHUNK_SIZE = 4096

# Initialize PyAudio
try:
    p = pyaudio.PyAudio()
    print("PyAudio initialized successfully")
    
    # Get device info
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    print(f"Number of audio devices: {numdevices}")
    
    # List all audio input devices
    print("\nAudio Input Devices:")
    for i in range(numdevices):
        device_info = p.get_device_info_by_index(i)
        if device_info.get('maxInputChannels') > 0:
            print(f"Device {i}: {device_info.get('name')}")
    
    # Try to open the default input stream
    print("\nAttempting to open audio stream...")
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=AUDIO_CHUNK_SIZE
    )
    
    print("Audio stream opened successfully")
    print("Recording for 3 seconds...")
    
    # Record for a few seconds
    for i in range(3):
        audio_data = stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        print(f"Audio chunk {i+1}: min={audio_np.min():.2f}, max={audio_np.max():.2f}, mean={audio_np.mean():.2f}")
        time.sleep(1)
    
    # Close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Audio test completed successfully")
    
except Exception as e:
    print(f"Error: {e}")