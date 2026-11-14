"""Pytest configuration and fixtures."""

import os
from pathlib import Path

import pytest

# Set up headless display for GUI tests
os.environ.setdefault("DISPLAY", ":99")

# pynput's X backend opens ~/.Xauthority at import; on a fresh CI runner
# the file doesn't exist and import fails with XauthError, cascading every
# test ERROR at collection. xvfb-run -a creates one per invocation, but
# pynput resolves the path via $XAUTHORITY (set by our autouse fixture
# below to /tmp/.Xauthority). Make sure the file exists so pynput can
# read it even before the fixture runs (it doesn't apply at import time).
try:
    Path("/tmp/.Xauthority").touch(exist_ok=True)
except OSError:
    pass


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

    # Create mock Xauthority file path
    monkeypatch.setenv("XAUTHORITY", "/tmp/.Xauthority")


@pytest.fixture
def sample_numpy_image():
    """Create a sample numpy array image for testing."""
    import numpy as np

    return np.ones((480, 640, 3), dtype=np.uint8) * 255


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path
