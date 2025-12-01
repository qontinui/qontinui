"""
Tauri runtime extractor.

Extracts UI state from Tauri applications using Playwright with Tauri API mocks.
"""

from .extractor import TauriExtractor
from .mock import (
    TAURI_MOCK_SCRIPT,
    create_fs_mock,
    generate_mock_script,
    get_default_mocks,
)

__all__ = [
    "TauriExtractor",
    "TAURI_MOCK_SCRIPT",
    "generate_mock_script",
    "get_default_mocks",
    "create_fs_mock",
]
