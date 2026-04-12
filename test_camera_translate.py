import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies that might be missing or slow to load
sys.modules['cv2'] = MagicMock()
sys.modules['mediapipe'] = MagicMock()
sys.modules['tsl_tasks_extractor'] = MagicMock()
import torch
sys.modules['torch'] = torch

from PIL import ImageFont
from camera_translate import ThaiSignTranslator

class TestFontCaching(unittest.TestCase):
    @patch('camera_translate.load_model')
    @patch('camera_translate.MediaPipeTasksLandmarkExtractor')
    def test_get_cached_font(self, mock_extractor, mock_load_model):
        mock_load_model.return_value = (MagicMock(), ['label1'], {0: 'label1'}, [], [])

        translator = ThaiSignTranslator(model_path="dummy_path.pt")

        # Test default font caching
        with patch('PIL.ImageFont.load_default') as mock_load_default:
            mock_font = MagicMock()
            mock_load_default.return_value = mock_font

            font1 = translator._get_cached_font(20)
            font2 = translator._get_cached_font(20)

            self.assertEqual(font1, font2)
            mock_load_default.assert_called_once_with(size=20)

        # Test truetype font caching
        with patch('PIL.ImageFont.truetype') as mock_truetype:
            mock_font = MagicMock()
            mock_truetype.return_value = mock_font

            font1 = translator._get_cached_font(30, "dummy_font.ttf")
            font2 = translator._get_cached_font(30, "dummy_font.ttf")

            self.assertEqual(font1, font2)
            mock_truetype.assert_called_once_with("dummy_font.ttf", 30)

if __name__ == '__main__':
    unittest.main()
