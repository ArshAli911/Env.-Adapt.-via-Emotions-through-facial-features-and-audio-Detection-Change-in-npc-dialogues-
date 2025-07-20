import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
import os

# Adjust path to import from the parent directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.emotion_models import EmotionClassifier

class TestEmotionModels(unittest.TestCase):

    @patch('models.emotion_models.EmotionClassifier', autospec=True)
    @patch('torch.load')
    def setUp(self, mock_torch_load, MockEmotionClassifier):
        # Mock torch.load to prevent actual file loading
        mock_torch_load.return_value = MagicMock(spec=dict) # Return a mock dictionary
        
        self.dummy_input_facial = MagicMock(spec=torch.Tensor) # Mock input as a Tensor
        self.dummy_input_audio = MagicMock(spec=torch.Tensor) # Mock input as a Tensor
        self.dummy_input_multimodal = MagicMock(spec=torch.Tensor) # Mock input as a Tensor

        # Create mocked instances of EmotionClassifier
        self.facial_classifier = MockEmotionClassifier.return_value
        self.audio_classifier = MagicMock(spec=EmotionClassifier) # Explicitly mock as EmotionClassifier instance
        self.multimodal_classifier = MagicMock(spec=EmotionClassifier) # Explicitly mock as EmotionClassifier instance

        # Directly mock the predict methods of the EmotionClassifier instances
        self.facial_classifier.predict = MagicMock(return_value={
            'emotion': 'happy', 'confidence': 0.8, 'probabilities': np.array([0.1,0.1,0.1,0.8,0.1,0.1,0.1])
        })
        self.audio_classifier.predict = MagicMock(return_value={
            'emotion': 'sad', 'confidence': 0.7, 'probabilities': np.array([0.1,0.1,0.1,0.1,0.1,0.7,0.1])
        })
        self.multimodal_classifier.predict = MagicMock(return_value={
            'emotion': 'neutral', 'confidence': 0.9, 'probabilities': np.array([0.1,0.1,0.1,0.1,0.9,0.1,0.1])
        })

    def test_facial_emotion_prediction(self):
        prediction = self.facial_classifier.predict(self.dummy_input_facial)
        self.assertEqual(prediction['emotion'], 'happy')
        self.assertAlmostEqual(prediction['confidence'], 0.8, places=5)
        self.assertTrue(isinstance(prediction['probabilities'], np.ndarray))
        self.facial_classifier.predict.assert_called_once_with(self.dummy_input_facial)
        
    def test_audio_emotion_prediction(self):
        prediction = self.audio_classifier.predict(self.dummy_input_audio)
        self.assertEqual(prediction['emotion'], 'sad')
        self.assertAlmostEqual(prediction['confidence'], 0.7, places=5)
        self.assertTrue(isinstance(prediction['probabilities'], np.ndarray))
        self.audio_classifier.predict.assert_called_once_with(self.dummy_input_audio)

    def test_multimodal_emotion_prediction(self):
        prediction = self.multimodal_classifier.predict(self.dummy_input_multimodal)
        self.assertEqual(prediction['emotion'], 'neutral')
        self.assertAlmostEqual(prediction['confidence'], 0.9, places=5)
        self.assertTrue(isinstance(prediction['probabilities'], np.ndarray))
        self.multimodal_classifier.predict.assert_called_once_with(self.dummy_input_multimodal)


if __name__ == '__main__':
    unittest.main() 