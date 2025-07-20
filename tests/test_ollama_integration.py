import unittest
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ollama_integration import OllamaIntegration

class TestOllamaIntegration(unittest.TestCase):

    def setUp(self):
        self.ollama_client = OllamaIntegration("deepseek-coder")

    def test_initialization(self):
        self.assertEqual(self.ollama_client.model, "deepseek-coder")

    @patch('ollama.chat')
    def test_analyze_text_emotion(self, mock_chat):
        mock_chat.return_value = {
            'message': {'content': 'happy'}
        }
        emotion = self.ollama_client.analyze_text_emotion("I am so joyful!")
        self.assertEqual(emotion, "happy")
        mock_chat.assert_called_once()

    @patch('models.ollama_integration.ollama.list', return_value={'models': [{'name': 'deepseek-coder:latest'}]})
    @patch('models.ollama_integration.OllamaIntegration.is_model_available', return_value=True)
    @patch('models.ollama_integration.ollama.generate')
    def test_generate_npc_dialogue(self, mock_generate, mock_available, mock_list):
        mock_generate.return_value = {
            'response': 'Hello there!'
        }
        dialogue = self.ollama_client.generate_npc_dialogue("neutral", context="meeting a new person")
        self.assertIn("Hello there!", dialogue)
        mock_generate.assert_called_once()
        self.assertEqual(len(self.ollama_client.dialogue_history), 1)
        self.assertEqual(self.ollama_client.dialogue_history[0]['role'], 'assistant')
        self.assertEqual(self.ollama_client.dialogue_history[0]['content'], 'Hello there!')

    @patch('models.ollama_integration.ollama.list', return_value={'models': [{'name': 'deepseek-coder:latest'}]})
    @patch('models.ollama_integration.OllamaIntegration.is_model_available', return_value=True)
    @patch('models.ollama_integration.ollama.generate')
    def test_dialogue_history(self, mock_generate, mock_available, mock_list):
        mock_generate.side_effect = [
            {'response': 'First response'},
            {'response': 'Second response'}
        ]
        self.ollama_client.generate_npc_dialogue("neutral")
        self.ollama_client.generate_npc_dialogue("happy")

        history = self.ollama_client.get_dialogue_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['content'], "First response")
        self.assertEqual(history[1]['content'], "Second response")

        self.ollama_client.clear_dialogue_history()
        self.assertEqual(len(self.ollama_client.dialogue_history), 0)

    @patch('models.ollama_integration.OllamaIntegration.is_model_available', return_value=False)
    def test_generate_npc_dialogue_model_unavailable(self, mock_available):
        dialogue = self.ollama_client.generate_npc_dialogue("neutral")
        self.assertEqual(dialogue, "Ollama model is not available for dialogue generation.")


if __name__ == '__main__':
    unittest.main() 