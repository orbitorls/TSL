import unittest
import sys
from unittest.mock import MagicMock, patch

# Mock all the heavy dependencies before importing the module under test
mock_torch = MagicMock()
mock_torch.nn = MagicMock()
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_torch.nn

mock_cv2 = MagicMock()
sys.modules['cv2'] = mock_cv2

mock_np = MagicMock()
sys.modules['numpy'] = mock_np

mock_pd = MagicMock()
sys.modules['pandas'] = mock_pd

mock_mediapipe = MagicMock()
sys.modules['mediapipe'] = mock_mediapipe

mock_pil = MagicMock()
mock_pil.ImageDraw = MagicMock()
mock_pil.ImageFont = MagicMock()
sys.modules['PIL'] = mock_pil

# Mock internal modules
mock_train_models = MagicMock()
sys.modules['src.train.models'] = mock_train_models

mock_data_extractor = MagicMock()
sys.modules['src.data.extractor'] = mock_data_extractor

# Now import the module under test
from src.inference.camera_translate import ThaiSignTranslator

class TestThaiSignTranslatorCache(unittest.TestCase):
    @patch('src.inference.camera_translate.resolve_feature_level_for_inference')
    @patch('src.inference.camera_translate.load_model')
    @patch('src.inference.camera_translate.ImageFont')
    def test_font_cache(self, mock_image_font, mock_load_model, mock_resolve_feature_level):
        # Setup mock load_model
        mock_load_model.return_value = (
            MagicMock(), ['A', 'B'], {0: 'A', 1: 'B'},
            MagicMock(), MagicMock(), True, 10, 'basic'
        )
        mock_resolve_feature_level.return_value = ('basic', None)

        # Setup mock ImageFont to return distinct strings based on input args
        # This allows us to verify cache hits vs misses properly
        def mock_truetype(path, size):
            return f"MockFont_{path}_{size}"

        mock_image_font.truetype.side_effect = mock_truetype

        # Initialize translator
        translator = ThaiSignTranslator("dummy_path")

        font_paths = ("C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/phagspa.ttf")
        font_size = 80

        # Call 1: Should be a cache miss, so it calls ImageFont.truetype
        font1 = translator._get_cached_font(font_size, font_paths)
        self.assertEqual(font1, "MockFont_C:/Windows/Fonts/tahoma.ttf_80")
        self.assertEqual(mock_image_font.truetype.call_count, 1)

        # Call 2: Should be a cache hit, so call count remains 1
        font2 = translator._get_cached_font(font_size, font_paths)
        self.assertEqual(font2, "MockFont_C:/Windows/Fonts/tahoma.ttf_80")
        self.assertEqual(mock_image_font.truetype.call_count, 1) # Verify no new call

        # Call 3: Different size, should be a cache miss
        font3 = translator._get_cached_font(30, font_paths)
        self.assertEqual(font3, "MockFont_C:/Windows/Fonts/tahoma.ttf_30")
        self.assertEqual(mock_image_font.truetype.call_count, 2)

        # Call 4: Cache hit for different size
        font4 = translator._get_cached_font(30, font_paths)
        self.assertEqual(font4, "MockFont_C:/Windows/Fonts/tahoma.ttf_30")
        self.assertEqual(mock_image_font.truetype.call_count, 2)

    @patch('src.inference.camera_translate.resolve_feature_level_for_inference')
    @patch('src.inference.camera_translate.load_model')
    @patch('src.inference.camera_translate.ImageFont')
    def test_font_cache_fallback(self, mock_image_font, mock_load_model, mock_resolve_feature_level):
        mock_load_model.return_value = (
            MagicMock(), ['A', 'B'], {0: 'A', 1: 'B'},
            MagicMock(), MagicMock(), True, 10, 'basic'
        )
        mock_resolve_feature_level.return_value = ('basic', None)

        # Truetype always fails
        mock_image_font.truetype.side_effect = Exception("Font not found")

        # Load default returns a predictable value
        mock_image_font.load_default.return_value = "MockDefaultFont"

        translator = ThaiSignTranslator("dummy_path")

        font_paths = ("C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/phagspa.ttf")

        # Call 1: Miss, tries truetype (fails), calls load_default
        font1 = translator._get_cached_font(80, font_paths)
        self.assertEqual(font1, "MockDefaultFont")
        self.assertEqual(mock_image_font.truetype.call_count, 2) # Tries both paths
        self.assertEqual(mock_image_font.load_default.call_count, 1)

        # Call 2: Hit, no new calls
        font2 = translator._get_cached_font(80, font_paths)
        self.assertEqual(font2, "MockDefaultFont")
        self.assertEqual(mock_image_font.truetype.call_count, 2)
        self.assertEqual(mock_image_font.load_default.call_count, 1)

if __name__ == '__main__':
    unittest.main()
