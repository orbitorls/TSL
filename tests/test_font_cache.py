import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies before importing the module
sys.modules['cv2'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['mediapipe'] = MagicMock()
sys.modules['PIL'] = MagicMock()

# Specifically mock ImageFont and its truetype function
mock_imagefont = MagicMock()

sys.modules['tsl_tasks_extractor'] = MagicMock()

import camera_translate
camera_translate.ImageFont = mock_imagefont

class TestFontCache(unittest.TestCase):
    def setUp(self):
        # Reset mocks before each test
        mock_imagefont.truetype.reset_mock()
        mock_imagefont.load_default.reset_mock()

        # Custom side effect to track unique font loads
        def truetype_side_effect(path, size, *args, **kwargs):
            # Return a string instead of a mock object to verify distinct cache hits
            return f"MockFont-{path}-{size}"
        mock_imagefont.truetype.side_effect = truetype_side_effect

        def load_default_side_effect(size=10, *args, **kwargs):
            return f"MockDefaultFont-{size}"
        mock_imagefont.load_default.side_effect = load_default_side_effect

        # Mock load_model to avoid file I/O errors during init
        with patch('camera_translate.load_model') as mock_load_model:
            mock_load_model.return_value = (MagicMock(), ["A"], {0: "A"}, [0.0], [1.0])
            self.translator = camera_translate.ThaiSignTranslator(model_path="dummy_path.pt")

    def test_cache_miss_and_hit(self):
        font_paths = ["fake/path.ttf"]
        size = 20

        # First call: Should be a cache miss and call ImageFont.truetype
        font1 = self.translator._get_cached_font(font_paths, size)
        self.assertEqual(font1, "MockFont-fake/path.ttf-20")
        mock_imagefont.truetype.assert_called_once_with("fake/path.ttf", 20)

        # Reset mock call count
        mock_imagefont.truetype.reset_mock()

        # Second call: Should be a cache hit and return the exact same object
        font2 = self.translator._get_cached_font(font_paths, size)
        self.assertEqual(font2, "MockFont-fake/path.ttf-20")

        # Verify ImageFont.truetype was NOT called again
        mock_imagefont.truetype.assert_not_called()

    def test_cache_default_font(self):
        # Test fallback when font path is empty/invalid
        size = 30

        # First call: Cache miss, should use default
        font1 = self.translator._get_cached_font(None, size)
        self.assertEqual(font1, "MockDefaultFont-30")
        mock_imagefont.load_default.assert_called_once_with(size=30)

        mock_imagefont.load_default.reset_mock()

        # Second call: Cache hit
        font2 = self.translator._get_cached_font(None, size)
        self.assertEqual(font2, "MockDefaultFont-30")
        mock_imagefont.load_default.assert_not_called()

    def test_cache_tuple_key(self):
        # Verify lists are correctly converted to tuples for caching
        paths1 = ["font1.ttf", "font2.ttf"]
        paths2 = ["font1.ttf", "font2.ttf"] # Different list object, same content
        size = 40

        font1 = self.translator._get_cached_font(paths1, size)
        mock_imagefont.truetype.assert_called_once()

        mock_imagefont.truetype.reset_mock()

        font2 = self.translator._get_cached_font(paths2, size)
        mock_imagefont.truetype.assert_not_called()
        self.assertEqual(font1, font2)

if __name__ == '__main__':
    unittest.main()
