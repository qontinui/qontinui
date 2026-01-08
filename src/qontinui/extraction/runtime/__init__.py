"""
Runtime extraction module.

Provides extractors for live applications across different runtime environments:
- Web applications (Playwright)
- Tauri applications (Playwright + Tauri API mocks)
- Electron applications (future)
- Native desktop applications (future)
"""

from .base import RuntimeExtractor
from .types import (
    ExtractionTarget,
    RuntimeExtractionSession,
    RuntimeStateCapture,
    RuntimeType,
)


def __getattr__(name: str):
    """Lazy import for Playwright-dependent extractors."""
    if name == "PlaywrightExtractor":
        from .playwright import PlaywrightExtractor

        return PlaywrightExtractor
    if name == "TauriExtractor":
        from .tauri import TauriExtractor

        return TauriExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base classes
    "RuntimeExtractor",
    # Extractors (lazy imports)
    "PlaywrightExtractor",
    "TauriExtractor",
    # Types
    "ExtractionTarget",
    "RuntimeStateCapture",
    "RuntimeExtractionSession",
    "RuntimeType",
]
