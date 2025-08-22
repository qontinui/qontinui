"""Model transition package - ported from Qontinui framework.

Core transition interfaces and enums.
"""

from .state_transition import StateTransition, StaysVisible

__all__ = [
    'StateTransition',
    'StaysVisible',
]