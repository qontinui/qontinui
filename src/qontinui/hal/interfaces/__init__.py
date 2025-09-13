"""HAL Interface definitions.

These interfaces define the contracts that all HAL implementations must follow.
"""

from .screen_capture import IScreenCapture
from .pattern_matcher import IPatternMatcher
from .input_controller import IInputController, MouseButton, Key
from .ocr_engine import IOCREngine, TextRegion, TextMatch
from .platform_specific import IPlatformSpecific

__all__ = [
    'IScreenCapture',
    'IPatternMatcher', 
    'IInputController',
    'MouseButton',
    'Key',
    'IOCREngine',
    'TextRegion',
    'TextMatch',
    'IPlatformSpecific'
]