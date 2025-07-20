import subprocess
import sys

def main():
    print("Starting Streamlit VR Emotion Adaptation application...")
    try:
        # Ensure streamlit is installed
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], check=True)
        # Run the Streamlit application
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except FileNotFoundError:
        print("Error: 'streamlit' command not found. Please ensure Streamlit is installed and in your PATH.")
        print("You can install it using: pip install streamlit")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 