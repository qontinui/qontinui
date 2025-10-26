"""Wrapper layer for routing mock vs real automation (Brobot pattern).

Provides wrapper classes that route automation calls to either mock
implementations (historical playback) or real HAL implementations
(actual GUI automation) based on ExecutionMode configuration.

This follows the Brobot pattern where high-level actions are agnostic
to whether they're running in mock or real mode - the wrapper layer
handles all routing decisions.
"""

from .base import ActionWrapper, BaseWrapper
from .capture_wrapper import CaptureWrapper
from .controller import ExecutionModeController, get_controller
from .find_wrapper import FindWrapper
from .input_wrapper import KeyboardWrapper, MouseWrapper
from .keyboard import Keyboard
from .mouse import Mouse
from .screen import Screen
from .time_wrapper import TimeWrapper

__all__ = [
    "ActionWrapper",
    "BaseWrapper",
    "FindWrapper",
    "CaptureWrapper",
    "MouseWrapper",
    "KeyboardWrapper",
    "Keyboard",
    "Mouse",
    "Screen",
    "TimeWrapper",
    "ExecutionModeController",
    "get_controller",
]
