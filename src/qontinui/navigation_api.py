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
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from qontinui.model.state.state_service import StateService
    from qontinui.multistate_integration.pathfinding_navigator import (
        PathfindingNavigator,
    )
    from qontinui.state_management.state_memory import StateMemory

logger = logging.getLogger(__name__)

# Global navigation instance
_navigator: Optional["PathfindingNavigator"] = None
_state_memory: Optional["StateMemory"] = None
_state_service: Optional["StateService"] = None
_workflow_executor: Any | None = None
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
        from qontinui.config.state_loader import load_states_from_config
        from qontinui.config.transition_loader import load_transitions_from_config
        from qontinui.model.state.state_service import StateService
        from qontinui.multistate_integration.pathfinding_navigator import (
            PathfindingNavigator,
        )
        from qontinui.state_management.state_memory import StateMemory

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

        # Debug: Check if transitions were actually added to states
        logger.debug("Checking states after transition loading:")
        for state in _state_service.get_all_states():
            logger.debug(f"State '{state.name}' has {len(state.transitions)} transitions")

        # Step 4: Initialize StateMemory with populated StateService
        _state_memory = StateMemory(state_service=_state_service)  # type: ignore[arg-type]
        logger.debug("StateMemory initialized")

        # Step 5: Activate initial states
        initial_states = []
        for state in _state_service.get_all_states():
            # States marked with is_initial = True are activated as starting states
            if state.is_initial and state.id is not None:
                initial_states.append(state.id)
                _state_memory.add_active_state(state.id)
                logger.debug(f"Activated initial state: {state.name} (ID: {state.id})")

        if initial_states:
            logger.info(f"Activated {len(initial_states)} initial state(s)")
        else:
            logger.warning(
                "No initial states found - navigation may require explicit state activation"
            )

        # Step 5.5: Initialize PathfindingNavigator WITHOUT a workflow_executor
        # The runner will call set_workflow_executor() later to inject the proper executor
        # that has access to all workflows and can execute them
        _workflow_executor = None
        logger.debug(
            "PathfindingNavigator will be created without workflow_executor - runner must call set_workflow_executor()"
        )

        # Step 6: Initialize PathfindingNavigator
        _navigator = PathfindingNavigator(_state_memory, workflow_executor=None)
        logger.info(
            "PathfindingNavigator initialized - workflow_executor will be set by runner after configuration loads"
        )
        logger.debug(
            "  (Note: workflow_executor is None until runner calls set_workflow_executor())"
        )

        # Step 7: Register all states with the multistate adapter
        registered_count = 0
        for state in _state_service.get_all_states():
            if state.id is not None:
                _navigator.multistate_adapter.register_qontinui_state(state)
                registered_count += 1

        logger.info(f"Registered {registered_count} states with multistate adapter")

        # Step 8: Register all transitions with the multistate adapter
        transition_count = 0
        states_with_transitions = 0
        for state in _state_service.get_all_states():
            if state.transitions:
                states_with_transitions += 1
                logger.info(f"State '{state.name}' has {len(state.transitions)} transitions")
            for transition in state.transitions:
                logger.info(f"Registering transition: {transition.id} from state '{state.name}'")  # type: ignore[attr-defined]
                _navigator.multistate_adapter.register_qontinui_transition(transition)  # type: ignore[arg-type]
                transition_count += 1

        logger.info(f"Found {states_with_transitions} states with transitions")
        logger.info(f"Registered {transition_count} total transitions with multistate adapter")

        _initialized = True

        logger.info(f"Navigation system initialized successfully with {state_count} states")
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
        logger.error("Navigation system not initialized - call load_configuration() first")
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


def open_states(state_identifiers: list[str | int]) -> bool:
    """Navigate to multiple specified states.

    This function finds a path that activates ALL specified target states.
    The library handles all pathfinding, state detection, and transition execution.

    Args:
        state_identifiers: List of state names (str) or state IDs (int) to navigate to.
                          All states must be reached for the navigation to succeed.

    Returns:
        True if navigation succeeded and ALL target states were reached, False otherwise
    """
    if not _initialized:
        logger.error("Navigation system not initialized - call load_configuration() first")
        return False

    if not _navigator:
        logger.error("Navigator not available")
        return False

    if not _state_service:
        logger.error(
            "State service not available - state/transition loading from config not yet implemented"
        )
        return False

    if not state_identifiers:
        logger.error("No state identifiers provided")
        return False

    # Convert all identifiers to state IDs
    target_state_ids = []
    for identifier in state_identifiers:
        if isinstance(identifier, int):
            state_id = identifier
            state = _state_service.get_state_by_id(state_id)  # type: ignore[attr-defined]
            if not state:
                logger.error(f"State ID {state_id} not found in state service")
                return False
        elif isinstance(identifier, str):
            state = _state_service.get_state_by_name(identifier)
            if not state or not state.id:
                logger.error(f"State '{identifier}' not found in state service")
                return False
            state_id = state.id
        else:
            logger.error(f"Invalid state identifier type: {type(identifier)}")
            return False

        target_state_ids.append(state_id)

    logger.info(f"Navigating to states: {state_identifiers} (IDs: {target_state_ids})")

    # Get current active states for debugging
    active_states = _state_memory.get_active_state_names() if _state_memory else []
    logger.info(f"Current active states: {active_states}")

    # Navigate using PathfindingNavigator with multiple targets
    context = _navigator.navigate_to_states(target_state_ids, execute=True)

    if context:
        logger.info(
            f"Navigation context: targets_reached={context.targets_reached}, "
            f"target_states={context.path.target_states if context.path else None}"
        )
    else:
        logger.error("Navigation returned no context - pathfinding failed")
        return False

    if context.targets_reached == context.path.target_states:
        logger.info(f"Successfully navigated to all target states: {state_identifiers}")
        return True
    else:
        logger.warning(f"Failed to navigate to all target states: {state_identifiers}")
        logger.warning(
            f"Targets reached: {context.targets_reached}, Expected: {context.path.target_states}"
        )
        return False


def set_workflow_executor(workflow_executor: Any) -> None:
    """Set the workflow executor for executing transition workflows.

    This should be called by the runner after load_configuration() to enable
    workflow execution during navigation.

    Args:
        workflow_executor: An executor that can run workflows (e.g., ActionExecutor)
    """
    global _workflow_executor, _navigator

    _workflow_executor = workflow_executor
    logger.info("Workflow executor set for navigation")

    # Update the navigator's transition executor if navigator exists
    if _navigator and _navigator.transition_executor:
        _navigator.transition_executor.workflow_executor = workflow_executor
        logger.info("Updated existing navigator with workflow executor")


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


def reset_to_initial_state() -> bool:
    """Reset navigation state to initial conditions.

    This function clears all active states and re-activates only the initial states.
    Should be called before each automation run to ensure consistent starting state.

    Returns:
        True if reset succeeded, False otherwise
    """
    global _state_memory, _state_service, _initialized

    if not _initialized:
        logger.warning("Navigation system not initialized - cannot reset state")
        return False

    if not _state_memory or not _state_service:
        logger.error("State memory or state service not available")
        return False

    try:
        logger.info("Resetting navigation state to initial conditions")

        # Clear all active states
        _state_memory.clear_active_states()
        logger.debug("Cleared all active states")

        # Re-activate initial states
        initial_state_count = 0
        for state in _state_service.get_all_states():
            if state.is_initial and state.id is not None:
                _state_memory.add_active_state(state.id)
                logger.debug(f"Re-activated initial state: {state.name} (ID: {state.id})")
                initial_state_count += 1

        logger.info(f"Reset complete - activated {initial_state_count} initial state(s)")
        return True

    except Exception as e:
        logger.error(f"Failed to reset navigation state: {e}", exc_info=True)
        return False
