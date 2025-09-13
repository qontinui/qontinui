"""State management package - ported from Qontinui framework.

This package handles all state-related functionality including:
- Tracking active and hidden states
- State detection and discovery
- State transitions and navigation
- Initial state determination
- High-level state automation
"""

from .state_memory import StateMemory, StateMemoryEnum, StateService
from .active_state_set import ActiveStateSet
from .adjacent_states import AdjacentStates
from .initial_states import InitialStates
from .state_detector import StateDetector
from .state_id_resolver import StateIdResolver
from .state_visibility_manager import StateVisibilityManager
from .search_region_dependency_initializer import SearchRegionDependencyInitializer
from .manager import QontinuiStateManager
from .state_automator import StateAutomator

__all__ = [
    'StateMemory',
    'StateMemoryEnum',
    'StateService',
    'ActiveStateSet',
    'AdjacentStates',
    'InitialStates',
    'StateDetector',
    'StateIdResolver',
    'StateVisibilityManager',
    'SearchRegionDependencyInitializer',
    'QontinuiStateManager',
    'StateAutomator',
]