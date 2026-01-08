"""
Native application extraction module using OS Accessibility APIs.

Extracts UI elements and states from native applications using:
- Windows: UI Automation (UIA)
- macOS: Accessibility API (AX)
- Linux: AT-SPI (Assistive Technology Service Provider Interface)

This module provides rich semantic information (roles, labels, states)
that may not be available through other extraction methods.

Usage:
    >>> from qontinui.extraction.native import AccessibilityExtractor
    >>> from qontinui.extraction.abstract_extractor import ExtractionContext
    >>> from qontinui.extraction.extractor_config import ExtractorConfig
    >>>
    >>> extractor = AccessibilityExtractor()
    >>> context = ExtractionContext(
    ...     app_name="Notepad",
    ...     platform="win32"
    ... )
    >>> config = ExtractorConfig()
    >>> result = await extractor.extract(context, config)

Architecture:
    native/
    |-- __init__.py              # This file
    |-- accessibility_extractor.py  # Main AccessibilityExtractor
    |-- base_api.py              # Platform API abstraction (TODO)
    |-- windows/                 # Windows UI Automation (TODO)
    |   |-- __init__.py
    |   |-- uia_api.py
    |   |-- uia_walker.py
    |-- macos/                   # macOS Accessibility (TODO)
    |   |-- __init__.py
    |   |-- ax_api.py
    |-- linux/                   # Linux AT-SPI (TODO)
        |-- __init__.py
        |-- atspi_api.py

Platform Support:
    - Windows: Full support via comtypes/pywinauto
    - macOS: Full support via pyobjc
    - Linux: Support via python-atspi/pyatspi2

Notes:
    - Requires appropriate platform libraries to be installed
    - May require elevated permissions on some systems
    - Performance varies by application complexity
"""

from .accessibility_extractor import AccessibilityExtractor

__all__ = [
    "AccessibilityExtractor",
]
