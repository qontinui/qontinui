"""Adapter implementations following Single Responsibility Principle.

Provides specialized adapters for different action types:
- MouseAdapter: Mouse-specific actions
- KeyboardAdapter: Keyboard-specific actions
- ScreenAdapter: Screen-specific actions
- AdapterResult: Common result type

Also exports the main facade adapters from parent module.
"""

from .adapter_result import AdapterResult
from .keyboard_adapter import HALKeyboardAdapter, KeyboardAdapter, SeleniumKeyboardAdapter
from .mouse_adapter import HALMouseAdapter, MouseAdapter, SeleniumMouseAdapter
from .screen_adapter import HALScreenAdapter, ScreenAdapter, SeleniumScreenAdapter

__all__ = [
    "AdapterResult",
    "MouseAdapter",
    "HALMouseAdapter",
    "SeleniumMouseAdapter",
    "KeyboardAdapter",
    "HALKeyboardAdapter",
    "SeleniumKeyboardAdapter",
    "ScreenAdapter",
    "HALScreenAdapter",
    "SeleniumScreenAdapter",
]
