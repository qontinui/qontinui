"""Navigation transition package - ported from Qontinui framework.

Contains state transition implementations and management.
"""

from .code_state_transition import CodeStateTransition
from .state_transitions import StateTransitions, StateTransitionsBuilder

__all__ = [
    "StateTransitions",
    "StateTransitionsBuilder",
    "CodeStateTransition",
]
