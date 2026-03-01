"""HAL implementation modules."""

# Screen capture implementations
# Input controller implementations (standalone keyboard and mouse)
from .keyboard_operations import KeyboardOperations
from .mouse_operations import MouseOperations
from .mss_capture import MSSScreenCapture

# Pattern matcher implementations
from .opencv_matcher import OpenCVMatcher

# PyAutoGUI implementations
from .pyautogui_keyboard import PyAutoGUIKeyboardOperations
from .pyautogui_mouse import PyAutoGUIMouseOperations


def __getattr__(name: str):  # noqa: N807
    """Lazy import for optional heavy dependencies (EasyOCR)."""
    if name == "EasyOCREngine":
        from .easyocr_engine import EasyOCREngine

        globals()["EasyOCREngine"] = EasyOCREngine
        return EasyOCREngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MSSScreenCapture",
    "OpenCVMatcher",
    "EasyOCREngine",
    "KeyboardOperations",
    "MouseOperations",
    "PyAutoGUIKeyboardOperations",
    "PyAutoGUIMouseOperations",
]
