import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock all heavy dependencies before importing
sys.modules['cv2'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['mediapipe'] = MagicMock()

# Mock PIL components properly
mock_pil_image = MagicMock()
mock_image_draw = MagicMock()
mock_image_font = MagicMock()
mock_pil = MagicMock()
mock_pil.Image = mock_pil_image
mock_pil.ImageDraw = mock_image_draw
mock_pil.ImageFont = mock_image_font
sys.modules['PIL'] = mock_pil

# Mock internal dependencies
mock_tsl_tasks = MagicMock()
mock_tsl_tasks.LANDMARK_VECTOR_DIM = 174
sys.modules['tsl_tasks_extractor'] = mock_tsl_tasks

# Now we can import the module safely
from camera_translate import ThaiSignTranslator, load_model

class TestFontCaching(unittest.TestCase):
    def setUp(self):
        # Reset mocks
        mock_image_font.truetype.reset_mock()
        mock_image_font.load_default.reset_mock()

    @patch('camera_translate.load_model')
    @patch('camera_translate.MediaPipeTasksLandmarkExtractor')
    def test_font_caching(self, mock_extractor, mock_load_model):
        # Setup mock return values for load_model
        mock_model = MagicMock()
        mock_labels = ["test"]
        mock_idx = {0: "test"}
        mock_mean = [0] * 174
        mock_std = [1] * 174
        mock_load_model.return_value = (mock_model, mock_labels, mock_idx, mock_mean, mock_std)

        # Setup mock for truetype to simulate successful load
        def mock_truetype(path, size):
            return f"font_obj_size_{size}"

        mock_image_font.truetype.side_effect = mock_truetype

        # Create instance
        translator = ThaiSignTranslator(model_path="dummy.pt")

        # Verify cache is initially empty
        self.assertEqual(len(translator._font_cache), 0)

        # First call should miss cache and load font
        font1 = translator._get_cached_font(80)
        self.assertEqual(font1, "font_obj_size_80")
        self.assertEqual(mock_image_font.truetype.call_count, 1)
        self.assertEqual(len(translator._font_cache), 1)
        self.assertEqual(translator._font_cache[80], "font_obj_size_80")

        # Second call with same size should hit cache
        font2 = translator._get_cached_font(80)
        self.assertEqual(font2, "font_obj_size_80")
        # Call count should NOT increase
        self.assertEqual(mock_image_font.truetype.call_count, 1)

        # Call with different size should miss cache
        font3 = translator._get_cached_font(30)
        self.assertEqual(font3, "font_obj_size_30")
        self.assertEqual(mock_image_font.truetype.call_count, 2)
        self.assertEqual(len(translator._font_cache), 2)

    @patch('camera_translate.load_model')
    @patch('camera_translate.MediaPipeTasksLandmarkExtractor')
    def test_font_fallback(self, mock_extractor, mock_load_model):
        # Setup mock return values for load_model
        mock_model = MagicMock()
        mock_labels = ["test"]
        mock_idx = {0: "test"}
        mock_mean = [0] * 174
        mock_std = [1] * 174
        mock_load_model.return_value = (mock_model, mock_labels, mock_idx, mock_mean, mock_std)

        # Force truetype to fail so it falls back to load_default
        mock_image_font.truetype.side_effect = Exception("Font not found")

        def mock_load_default(*args, **kwargs):
            size = kwargs.get('size', 'default')
            return f"default_font_{size}"

        mock_image_font.load_default.side_effect = mock_load_default

        # Create instance
        translator = ThaiSignTranslator(model_path="dummy.pt")

        # Get font
        font = translator._get_cached_font(50)

        # Should fallback to default
        self.assertEqual(font, "default_font_50")
        self.assertEqual(translator._font_cache[50], "default_font_50")
        self.assertTrue(mock_image_font.load_default.called)

if __name__ == '__main__':
    unittest.main()
