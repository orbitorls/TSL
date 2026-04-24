import unittest
from unittest.mock import patch, MagicMock

# We must mock dependencies before importing camera_translate
import sys
mock_torch = MagicMock()
mock_cv2 = MagicMock()
mock_mediapipe = MagicMock()
mock_numpy = MagicMock()
mock_pil = MagicMock()

sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = MagicMock()
sys.modules['cv2'] = mock_cv2
sys.modules['mediapipe'] = mock_mediapipe
sys.modules['numpy'] = mock_numpy
sys.modules['PIL'] = mock_pil

# Provide required constants/methods for tsl_tasks_extractor
mock_extractor = MagicMock()
mock_extractor.LANDMARK_VECTOR_DIM = 162
mock_extractor.MediaPipeTasksLandmarkExtractor = MagicMock
mock_extractor.draw_debug_overlay = MagicMock()
mock_extractor.extract_features = MagicMock()
mock_extractor.normalize_features = MagicMock()
mock_extractor.report_extractor_compatibility = MagicMock()
sys.modules['tsl_tasks_extractor'] = mock_extractor

import camera_translate

class TestFontCache(unittest.TestCase):
    def setUp(self):
        # We need to mock `load_model` since it's called in `__init__`
        self.patcher_load_model = patch('camera_translate.load_model')
        self.mock_load_model = self.patcher_load_model.start()
        # Return dummy model components
        self.mock_load_model.return_value = (MagicMock(), ['a', 'b'], {0: 'a', 1: 'b'}, [0.0]*162, [1.0]*162)

    def tearDown(self):
        self.patcher_load_model.stop()

    @patch('camera_translate.ImageFont.truetype')
    @patch('camera_translate.ImageFont.load_default')
    def test_font_caching(self, mock_load_default, mock_truetype):
        # We need a unique identifiable return value to ensure cache hit vs miss
        # As per memory, using side_effect with a custom function
        def side_effect_truetype(path, size):
            return f"MockedFont-{size}"
        mock_truetype.side_effect = side_effect_truetype

        translator = camera_translate.ThaiSignTranslator("dummy_path")

        # Get font size 40 (First time)
        font_40 = translator._get_cached_font(40)
        self.assertEqual(font_40, "MockedFont-40")
        call_count_first = mock_truetype.call_count
        self.assertGreater(call_count_first, 0) # At least one path should be tried

        mock_truetype.reset_mock()

        # Get font size 40 (Second time - should be cached)
        font_40_cached = translator._get_cached_font(40)
        self.assertEqual(font_40_cached, "MockedFont-40")
        self.assertEqual(mock_truetype.call_count, 0) # Should not call truetype again

        # Get font size 60 (Different size - should not be cached)
        font_60 = translator._get_cached_font(60)
        self.assertEqual(font_60, "MockedFont-60")
        self.assertGreater(mock_truetype.call_count, 0) # Should call truetype again

if __name__ == '__main__':
    unittest.main()
