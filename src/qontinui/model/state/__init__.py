"""State package - ported from Qontinui framework.

Complete state management system following Brobot's architecture.
"""

from ..transition import StateTransition, StateTransitions
from .action_history import ActionHistory
from .action_snapshot import ActionSnapshot, ActionType, MatchResult
from .initial_states import (
    InitialStates,
    get_initial_states,
    register_initial_state,
    register_initial_states,
)
from .path import Path
from .path_finder import PathFinder
from .special import NullState, NullStateName, UnknownState, UnknownStateEnum
from .state import State, StateBuilder
from .state_cache_manager import StateCacheManager
from .state_enum import StateEnum
from .state_image import StateImage
from .state_lifecycle_manager import StateLifecycleManager, StateStatus
from .state_location import StateLocation
from .state_memory import StateMemory
from .state_metadata_tracker import StateMetadata, StateMetadataTracker
from .state_object import StateObject
from .state_object_metadata import StateObjectMetadata, StateObjectType
from .state_region import StateRegion
from .state_relationship_manager import StateRelationshipManager
from .state_repository import StateRepository
from .state_service import StateService
from .state_store import StateStore
from .state_string import StateString
from .state_validator import StateValidator

__all__ = [
    "State",
    "StateBuilder",
    "StateEnum",
    "StateImage",
    "StateObject",
    "StateLocation",
    "StateRegion",
    "StateString",
    "StateObjectMetadata",
    "StateObjectType",
    "StateTransition",
    "StateTransitions",
    "StateMemory",
    "StateService",
    "Path",
    "PathFinder",
    # State store components
    "StateStore",
    "StateRepository",
    "StateLifecycleManager",
    "StateStatus",
    "StateRelationshipManager",
    "StateMetadataTracker",
    "StateMetadata",
    "StateCacheManager",
    "StateValidator",
    # Action history and snapshots
    "ActionHistory",
    "ActionSnapshot",
    "ActionType",
    "MatchResult",
    # Special states
    "NullState",
    "NullStateName",
    "UnknownState",
    "UnknownStateEnum",
    # Initial states
    "InitialStates",
    "get_initial_states",
    "register_initial_state",
    "register_initial_states",
]
