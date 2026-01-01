"""
Runtime extraction module.

Provides extractors for live applications across different runtime environments:
- Web applications (Playwright)
- Tauri applications (Playwright + Tauri API mocks)
- Electron applications (future)
- Native desktop applications (future)
"""

from .base import RuntimeExtractor
from .playwright import PlaywrightExtractor
from .tauri import TauriExtractor
from .types import (
    ExtractionTarget,
    RuntimeExtractionSession,
    RuntimeStateCapture,
    RuntimeType,
)

__all__ = [
    # Base classes
    "RuntimeExtractor",
    # Extractors
    "PlaywrightExtractor",
    "TauriExtractor",
    # Types
    "ExtractionTarget",
    "RuntimeStateCapture",
    "RuntimeExtractionSession",
    "RuntimeType",
]
