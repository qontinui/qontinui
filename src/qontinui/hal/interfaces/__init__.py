"""HAL Interface definitions.

These interfaces define the contracts that all HAL implementations must follow.
"""

from .input_controller import IInputController, Key, MouseButton, MousePosition
from .keyboard_controller import IKeyboardController
from .mouse_controller import IMouseController
from .ocr_engine import IOCREngine, TextMatch, TextRegion
from .pattern_matcher import IPatternMatcher
from .platform_specific import IPlatformSpecific
from .screen_capture import IScreenCapture

__all__ = [
    "IScreenCapture",
    "IPatternMatcher",
    "IInputController",
    "IKeyboardController",
    "IMouseController",
    "MouseButton",
    "MousePosition",
    "Key",
    "IOCREngine",
    "TextRegion",
    "TextMatch",
    "IPlatformSpecific",
]
