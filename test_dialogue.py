#!/usr/bin/env python3
"""
Test dialogue generation with the updated Ollama integration
"""
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dialogue_generation():
    """Test the dialogue generation with different emotions"""
    print("🎭 Testing VR Emotion Adaptation Dialogue Generation...")
    
    try:
        from models.ollama_integration import OllamaIntegration
        from vr_adaptation.environment_controller import EnvironmentController
        
        print("✅ Successfully imported modules")
        
        # Test Ollama integration directly
        print("\n🔧 Testing Ollama Integration directly...")
        ollama_client = OllamaIntegration(model="deepseek-r1:latest")
        
        if not ollama_client.is_model_available():
            print("❌ Ollama model not available")
            return False
        
        print("✅ Ollama model is available")
        
        # Test different emotions
        emotions_to_test = ["happy", "sad", "angry", "fear", "surprise", "neutral"]
        
        print("\n🎯 Testing dialogue generation for different emotions:")
        for emotion in emotions_to_test:
            try:
                context = f"User is expressing {emotion} emotion in VR environment"
                dialogue = ollama_client.generate_npc_dialogue(emotion, context)
                
                print(f"  {emotion.upper()}: '{dialogue}'")
                
                if not dialogue or "error" in dialogue.lower():
                    print(f"    ⚠️  Warning: Generated dialogue might have issues")
                else:
                    print(f"    ✅ Success")
                    
            except Exception as e:
                print(f"    ❌ Error generating dialogue for {emotion}: {e}")
        
        # Test Environment Controller
        print("\n🏗️  Testing Environment Controller...")
        env_controller = EnvironmentController()
        
        test_dialogue = env_controller.generate_dialogue("happy")
        print(f"Environment Controller dialogue: '{test_dialogue}'")
        
        if test_dialogue and "error" not in test_dialogue.lower():
            print("✅ Environment Controller dialogue generation working")
            return True
        else:
            print("❌ Environment Controller dialogue generation failed")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dialogue_generation()
    
    if success:
        print("\n🎉 Dialogue generation is working correctly!")
        print("✅ Your VR Emotion Adaptation app should now generate NPC dialogue!")
    else:
        print("\n❌ Dialogue generation test failed")
        print("💡 Check the error messages above for troubleshooting")
        sys.exit(1)