import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock dependencies before importing
sys.modules['numpy'] = MagicMock()
sys.modules['pandas'] = MagicMock()
import numpy as np

mock_torch = MagicMock()
mock_torch.tensor = MagicMock(return_value=MagicMock())
mock_torch.no_grad = MagicMock(return_value=MagicMock())
mock_torch.softmax = MagicMock(return_value=MagicMock())
mock_torch.topk = MagicMock(return_value=MagicMock())
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = MagicMock()

mock_cv2 = MagicMock()
sys.modules['cv2'] = mock_cv2
sys.modules['mediapipe'] = MagicMock()

# Mock PIL components properly
import types
mock_pil = types.ModuleType('PIL')
mock_pil_image = MagicMock()
mock_pil_imagedraw = MagicMock()
mock_pil_imagefont = MagicMock()

mock_pil.Image = mock_pil_image
mock_pil.ImageDraw = mock_pil_imagedraw
mock_pil.ImageFont = mock_pil_imagefont

sys.modules['PIL'] = mock_pil
sys.modules['PIL.Image'] = mock_pil_image
sys.modules['PIL.ImageDraw'] = mock_pil_imagedraw
sys.modules['PIL.ImageFont'] = mock_pil_imagefont

from src.inference.camera_translate import ThaiSignTranslator


class TestThaiSignTranslatorCache(unittest.TestCase):
    @patch('src.inference.camera_translate.load_model')
    def setUp(self, mock_load_model):
        # Setup mock return for load_model
        mock_model = MagicMock()
        mock_labels = ['hello', 'world']
        mock_idx_to_label = {0: 'hello', 1: 'world'}
        mock_mean = np.zeros(162)
        mock_std = np.ones(162)
        mock_seq_mode = False
        mock_target_frames = 1
        mock_feature_level = 'basic'

        mock_load_model.return_value = (
            mock_model, mock_labels, mock_idx_to_label,
            mock_mean, mock_std, mock_seq_mode,
            mock_target_frames, mock_feature_level
        )

        self.translator = ThaiSignTranslator(model_path="dummy_path.pt")

    @patch('PIL.ImageFont.truetype')
    @patch('PIL.ImageFont.load_default')
    def test_get_cached_font_caches_result(self, mock_load_default, mock_truetype):
        # Setup mock behavior to return unique objects each time
        def side_effect_truetype(path, size):
            return f"MockFont_{path}_{size}"
        mock_truetype.side_effect = side_effect_truetype

        # Call multiple times with the same arguments
        font_paths = ["path/to/font.ttf"]

        font1 = self.translator._get_cached_font(42, font_paths)
        font2 = self.translator._get_cached_font(42, font_paths)
        font3 = self.translator._get_cached_font(42, font_paths)

        # Verify truetype was only called once
        self.assertEqual(mock_truetype.call_count, 1)
        mock_truetype.assert_called_with("path/to/font.ttf", 42)

        # Verify the same cached object is returned
        self.assertEqual(font1, font2)
        self.assertEqual(font2, font3)
        self.assertEqual(font1, "MockFont_path/to/font.ttf_42")

    @patch('PIL.ImageFont.truetype')
    @patch('PIL.ImageFont.load_default')
    def test_get_cached_font_handles_fallback(self, mock_load_default, mock_truetype):
        # Setup mock behavior: truetype fails, load_default succeeds
        mock_truetype.side_effect = Exception("Font not found")

        def side_effect_load_default(size=None):
            return f"DefaultFont_{size}"
        mock_load_default.side_effect = side_effect_load_default

        font_paths = ["invalid/path.ttf"]

        font1 = self.translator._get_cached_font(30, font_paths)
        font2 = self.translator._get_cached_font(30, font_paths)

        # Verify truetype was attempted but failed, and load_default was called once
        self.assertEqual(mock_truetype.call_count, 1)
        self.assertEqual(mock_load_default.call_count, 1)

        # Verify cache works for the fallback font
        self.assertEqual(font1, font2)
        self.assertEqual(font1, "DefaultFont_30")

if __name__ == '__main__':
    unittest.main()
