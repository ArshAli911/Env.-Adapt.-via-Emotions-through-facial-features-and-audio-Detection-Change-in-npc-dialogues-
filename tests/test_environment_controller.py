import unittest
from unittest.mock import MagicMock, patch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vr_adaptation.environment_controller import EnvironmentController, EmotionType
from models.ollama_integration import OllamaIntegration

class TestEnvironmentController(unittest.TestCase):

    @patch('models.ollama_integration.OllamaIntegration') # Corrected patching target
    def setUp(self, MockOllamaIntegration):
        self.mock_ollama_client_instance = MockOllamaIntegration.return_value
        self.controller = EnvironmentController()
        # Ensure ollama_client is the mocked instance
        self.controller.ollama_client = self.mock_ollama_client_instance

    def test_initialization(self):
        self.assertIsNotNone(self.controller.ollama_client)
        self.assertEqual(self.controller.current_emotion, EmotionType.NEUTRAL)
        self.assertEqual(self.controller.emotion_confidence, 0.0)

    def test_update_emotion(self):
        self.controller.update_emotion('happy', 0.8)
        self.assertEqual(self.controller.current_emotion, EmotionType.HAPPY)
        self.assertEqual(self.controller.emotion_confidence, 0.8)

        self.controller.update_emotion('sad', 0.5)
        self.assertEqual(self.controller.current_emotion, EmotionType.SAD)
        self.assertEqual(self.controller.emotion_confidence, 0.5)

    def test_update_emotion_invalid(self):
        # Test with an invalid emotion string (should default to neutral)
        self.controller.update_emotion('unknown', 0.7)
        self.assertEqual(self.controller.current_emotion, EmotionType.NEUTRAL)
        self.assertEqual(self.controller.emotion_confidence, 0.0)

    def test_generate_npc_dialogue_with_ollama(self):
        self.mock_ollama_client_instance.generate_npc_dialogue.return_value = "NPC says something happy."
        
        # Set a current emotion to test dialogue generation based on it
        self.controller.update_emotion('happy', 0.9)
        
        dialogue = self.controller.generate_dialogue("happy")
        self.assertEqual(dialogue, "NPC says something happy.")
        self.mock_ollama_client_instance.generate_npc_dialogue.assert_called_once() # Check if chat was requested

    def test_generate_npc_dialogue_with_ollama_no_response(self):
        self.mock_ollama_client_instance.generate_npc_dialogue.return_value = None
        self.controller.update_emotion('sad', 0.7)
        dialogue = self.controller.generate_dialogue("sad")
        self.assertIn("i'm not sure how to respond", dialogue.lower())
        self.mock_ollama_client_instance.generate_npc_dialogue.assert_called_once()

    def test_dialogue_generation_prompts(self):
        # Verify that the prompts passed to OllamaClient are correctly formed
        self.controller.update_emotion('fear', 0.8)
        self.mock_ollama_client_instance.generate_npc_dialogue.return_value = "Fearful response"
        self.controller.generate_dialogue("fear")
        
        # Get the arguments passed to the mock call
        args, kwargs = self.mock_ollama_client_instance.generate_npc_dialogue.call_args
        
        # Check the user prompt, specifically for emotion and context
        user_prompt = args[0] # The prompt is built within generate_npc_dialogue, emotion is the first positional argument
        self.assertEqual(user_prompt, "fear") # Only emotion is directly passed here


if __name__ == '__main__':
    unittest.main() 