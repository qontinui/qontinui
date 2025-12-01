"""
Injection scripts for runtime extraction.

Contains JavaScript injection scripts for mocking various runtime APIs.
"""

from pathlib import Path

# Path to Tauri mock script
TAURI_MOCK_JS_PATH = Path(__file__).parent / "tauri_mock.js"


def get_tauri_mock_script() -> str:
    """
    Get the Tauri mock script contents.

    Returns:
        JavaScript code for Tauri API mocking.
    """
    with open(TAURI_MOCK_JS_PATH, encoding="utf-8") as f:
        return f.read()


__all__ = [
    "TAURI_MOCK_JS_PATH",
    "get_tauri_mock_script",
]
