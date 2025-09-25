"""Model transition package - ported from Qontinui framework.

Core transition interfaces and enums.
"""

from .state_transition import StateTransition, StaysVisible
from .state_transitions import StateTransitions
from .transition_function import TransitionType

__all__ = [
    "StateTransition",
    "StateTransitions",
    "StaysVisible",
    "TransitionType",
]
