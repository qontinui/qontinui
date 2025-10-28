"""State management package - ported from Qontinui framework.

This package handles all state-related functionality including:
- Tracking active and hidden states
- State detection and discovery
- State transitions and navigation
- Initial state determination
- High-level state automation
"""

from .active_state_set import ActiveStateSet
from .adjacent_states import AdjacentStates
from .initial_states import InitialStates
from .manager import QontinuiStateManager
from .search_region_dependency_initializer import SearchRegionDependencyInitializer
from .state_automator import StateAutomator
from .state_detector import StateDetector
from .state_id_resolver import StateIdResolver
from .state_memory import StateMemory, StateMemoryEnum, StateService
from .state_visibility_manager import StateVisibilityManager

# Import model classes for convenience
try:
    from .models import Element, State, Transition
except ImportError:
    # Models may not be available in all configurations
    pass

__all__ = [
    "StateMemory",
    "StateMemoryEnum",
    "StateService",
    "ActiveStateSet",
    "AdjacentStates",
    "InitialStates",
    "StateDetector",
    "StateIdResolver",
    "StateVisibilityManager",
    "SearchRegionDependencyInitializer",
    "QontinuiStateManager",
    "StateAutomator",
    # Model classes (if available)
    "Element",
    "State",
    "Transition",
]
