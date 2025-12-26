"""Mock module for Qontinui - Brobot-style mocking implementation.

Note: MockFind functionality has been consolidated into
actions/find/mock_find_implementation.py which is used by FindAction.
"""

from .mock_action_history_factory import MockActionHistoryFactory
from .mock_actions import MockActions
from .mock_capture import MockCapture
from .mock_keyboard import MockKeyboard
from .mock_mode_manager import MockModeManager
from .mock_mouse import MockMouse
from .mock_state_management import MockStateManagement
from .mock_time import MockTime
from .recorder import RecorderConfig, SnapshotRecorder
from .snapshot import ActionHistory, ActionRecord

__all__ = [
    "MockStateManagement",
    "MockModeManager",
    "MockActionHistoryFactory",
    "MockActions",
    "MockCapture",
    "MockMouse",
    "MockKeyboard",
    "MockTime",
    "ActionRecord",
    "ActionHistory",
    "SnapshotRecorder",
    "RecorderConfig",
]
