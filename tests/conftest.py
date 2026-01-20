"""Pytest configuration and fixtures."""

import os
import sys
from unittest.mock import MagicMock

import pytest

# Set up headless display for GUI tests
os.environ.setdefault("DISPLAY", ":99")

# Mock modules that may not be available or have DLL issues
# This allows accessibility tests to run without full qontinui dependencies
sys.modules["pyautogui"] = MagicMock()
sys.modules["mouseinfo"] = MagicMock()
sys.modules["pyscreeze"] = MagicMock()


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "cdp_required: marks tests as requiring CDP (Chrome DevTools Protocol)",
    )


@pytest.fixture(autouse=True)
def mock_display_environment(monkeypatch):
    """Mock display environment for headless testing."""
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setenv("PYAUTOGUI_HEADLESS", "1")

    # Create mock Xauthority file path
    monkeypatch.setenv("XAUTHORITY", "/tmp/.Xauthority")


@pytest.fixture
def mock_pyautogui():
    """Provide a mock PyAutoGUI module."""
    mock = MagicMock()
    mock.FAILSAFE = True
    mock.PAUSE = 0.1
    mock.size.return_value = (1920, 1080)
    mock.position.return_value = (100, 100)
    mock.screenshot.return_value = MagicMock()

    return mock


@pytest.fixture
def sample_numpy_image():
    """Create a sample numpy array image for testing."""
    import numpy as np

    return np.ones((480, 640, 3), dtype=np.uint8) * 255


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path
