"""High-level navigation API for qontinui - to be called by the runner.

This module provides a simple API that the runner can use without needing to understand
the internal state management, pathfinding, or navigation details.

IMPORTANT ARCHITECTURE NOTE:
The qontinui library is responsible for ALL state management, including:
- Loading states and transitions from configuration
- Building the state graph
- Managing active states
- Finding paths between states
- Executing transitions

The runner should ONLY call the public API functions in this module.
The runner should NOT create StateMemory, Navigator, or any other state management objects.
"""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Global navigation instance
_navigator: Optional["PathfindingNavigator"] = None
_state_memory: Optional["StateMemory"] = None
_state_service: Optional["StateService"] = None
_initialized = False


def load_configuration(config_dict: dict[str, Any]) -> bool:
    """Load states and transitions from configuration.

    This is the main initialization function that should be called after
    the runner loads the JSON config file.

    The loading process follows these steps:
    1. Create StateService to manage all states
    2. Load states from config_dict["states"] using state_loader
    3. Load transitions from config_dict["transitions"] using transition_loader
    4. Initialize StateMemory with the populated StateService
    5. Initialize PathfindingNavigator for state navigation

    Note: Images and workflows must already be registered in the registry
    before calling this function.

    Args:
        config_dict: Complete configuration dictionary including states, transitions, etc.

    Returns:
        True if configuration loaded successfully, False otherwise

    Example:
        >>> from qontinui import navigation_api, registry
        >>> # Register images and workflows first
        >>> registry.register_image("img-1", my_image)
        >>> registry.register_workflow("wf-1", my_workflow)
        >>> # Then load configuration
        >>> success = navigation_api.load_configuration(config)
        >>> if success:
        ...     navigation_api.open_state("Target State")
    """
    global _navigator, _state_memory, _state_service, _initialized

    logger.info("Loading qontinui configuration...")

    try:
        # Import required modules
        from qontinui.model.state.state_service import StateService
        from qontinui.config.state_loader import load_states_from_config
        from qontinui.config.transition_loader import load_transitions_from_config
        from qontinui.state_management.state_memory import StateMemory
        from qontinui.multistate_integration.pathfinding_navigator import PathfindingNavigator

        # Step 1: Create StateService
        _state_service = StateService()
        logger.debug("StateService created")

        # Step 2: Load states from config
        if not load_states_from_config(config_dict, _state_service):
            logger.error("Failed to load states from configuration")
            return False

        state_count = len(_state_service.get_all_states())
        logger.info(f"Loaded {state_count} states from configuration")

        # Step 3: Load transitions from config
        if not load_transitions_from_config(config_dict, _state_service):
            logger.error("Failed to load transitions from configuration")
            return False

        logger.info("Loaded transitions from configuration")

        # Step 4: Initialize StateMemory with populated StateService
        _state_memory = StateMemory(state_service=_state_service)
        logger.debug("StateMemory initialized")

        # Step 5: Initialize PathfindingNavigator
        _navigator = PathfindingNavigator(_state_memory)
        logger.debug("PathfindingNavigator initialized")

        _initialized = True

        logger.info(
            f"Navigation system initialized successfully with {state_count} states"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        _initialized = False
        return False


def open_state(state_name: str) -> bool:
    """Navigate to the specified state by name.

    This is the main entry point for the runner to request state navigation.
    The library handles all pathfinding, state detection, and transition execution.

    Args:
        state_name: Name of the state to navigate to

    Returns:
        True if navigation succeeded, False otherwise
    """
    if not _initialized:
        logger.error(
            "Navigation system not initialized - call load_configuration() first"
        )
        return False

    if not _navigator:
        logger.error("Navigator not available")
        return False

    if not _state_service:
        logger.error(
            "State service not available - state/transition loading from config not yet implemented"
        )
        return False

    # Look up state ID from name
    state = _state_service.get_state_by_name(state_name)
    if not state or not state.id:
        logger.error(f"State '{state_name}' not found in state service")
        return False

    state_id = state.id
    logger.info(f"Navigating to state '{state_name}' (ID: {state_id})")

    # Navigate using PathfindingNavigator
    context = _navigator.navigate_to_states([state_id], execute=True)

    if context and context.targets_reached == context.path.target_states:
        logger.info(f"Successfully navigated to state '{state_name}'")
        return True
    else:
        logger.warning(f"Failed to navigate to state '{state_name}'")
        return False


def get_active_states() -> list[str]:
    """Get names of currently active states.

    Returns:
        List of active state names
    """
    if not _state_memory:
        return []

    return _state_memory.get_active_state_names()


def is_state_active(state_name: str) -> bool:
    """Check if a state is currently active.

    Args:
        state_name: State name to check

    Returns:
        True if state is active
    """
    if not _state_memory:
        return False

    return _state_memory.is_state_active(state_name)
