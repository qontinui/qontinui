"""Accessibility capture implementations.

This module provides implementations of the IAccessibilityCapture interface
for different accessibility backends:

- CDPAccessibilityCapture: Chrome DevTools Protocol (web browsers, Electron, Tauri on Windows)
- UIAAccessibilityCapture: Windows UI Automation (native Windows apps) [Windows only]
"""

import platform

from qontinui.hal.implementations.accessibility.cdp_capture import (
    CDPAccessibilityCapture,
)
from qontinui.hal.implementations.accessibility.ref_manager import RefManager

__all__ = [
    "CDPAccessibilityCapture",
    "RefManager",
]

# Conditionally export UIA capture on Windows
if platform.system() == "Windows":
    try:
        from qontinui.hal.implementations.accessibility.uia_capture import (  # noqa: F401
            UIAAccessibilityCapture,
        )

        __all__.append("UIAAccessibilityCapture")
    except ImportError:
        # uiautomation module not installed
        pass
