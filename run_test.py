import sys
import io
from unittest.mock import MagicMock
from src.train.compat import setup_windows_encoding

def test_setup_windows_encoding_noops_outside_windows():
    mock_stdout = MagicMock()

    # Store original state to verify
    orig_platform = sys.platform
    orig_stdout = sys.stdout

    try:
        sys.platform = "linux"
        sys.stdout = mock_stdout

        setup_windows_encoding()

        assert sys.stdout is mock_stdout, f"Expected {mock_stdout}, got {sys.stdout}"
        print("Test passed!")
    finally:
        sys.platform = orig_platform
        sys.stdout = orig_stdout

test_setup_windows_encoding_noops_outside_windows()
