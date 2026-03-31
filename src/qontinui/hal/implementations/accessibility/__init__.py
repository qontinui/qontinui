"""Accessibility capture implementations.

This module provides implementations of the IAccessibilityCapture interface
for different accessibility backends:

- CDPAccessibilityCapture: Chrome DevTools Protocol (web browsers, Electron, Tauri on Windows)
- UIAAccessibilityCapture: Windows UI Automation (native Windows apps) [Windows only]
- ATSPIAccessibilityCapture: AT-SPI2 (native Linux desktop apps) [Linux only]
- RustBackendCapture: Delegates to Rust runner's native accessibility layer via HTTP
"""

import platform

from qontinui.hal.implementations.accessibility.cdp_capture import (
    CDPAccessibilityCapture,
)
from qontinui.hal.implementations.accessibility.ref_manager import RefManager
from qontinui.hal.implementations.accessibility.rust_backend import (
    RustBackendCapture,
)

__all__ = [
    "CDPAccessibilityCapture",
    "RefManager",
    "RustBackendCapture",
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

# Conditionally export AT-SPI capture on Linux
if platform.system() == "Linux":
    try:
        from qontinui.hal.implementations.accessibility.atspi_capture import (  # noqa: F401
            ATSPIAccessibilityCapture,
        )

        __all__.append("ATSPIAccessibilityCapture")
    except ImportError:
        # pyatspi2 / gi.repository.Atspi not installed
        pass
