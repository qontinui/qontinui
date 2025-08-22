"""Special states package - ported from Qontinui framework.

Special states for handling edge cases in state management.
"""

from .null_state import NullState, NullStateName
from .unknown_state import UnknownState, UnknownStateEnum
from .special_state_type import SpecialStateType
from .state_text import StateText, TextMatchType, TextPattern

__all__ = [
    "NullState",
    "NullStateName",
    "UnknownState", 
    "UnknownStateEnum",
    "SpecialStateType",
    "StateText",
    "TextMatchType",
    "TextPattern",
]