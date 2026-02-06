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

from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qontinui.model.state.state_service import StateService
    from qontinui.multistate_integration.pathfinding_navigator import (
        PathfindingNavigator,
    )
    from qontinui.state_management.state_memory import StateMemory

logger = logging.getLogger(__name__)

_DEBUG_LOG_PATH = os.path.join(tempfile.gettempdir(), "qontinui_navigation_debug.log")


def _debug_print(msg: str) -> None:
    """Write debug message to file to ensure visibility when logging is disabled."""
    try:
        from qontinui_schemas.common import utc_now

        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            timestamp = utc_now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            f.write(f"[{timestamp}] [NAVIGATION_API] {msg}\n")
            f.flush()
    except Exception:
        pass


# Global navigation instance
_navigator: PathfindingNavigator | None = None
_state_memory: StateMemory | None = None
_state_service: StateService | None = None
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

    _debug_print("load_configuration() called")
    logger.info("Loading qontinui configuration...")

    try:
        # Import required modules
        _debug_print("Importing required modules...")
        from qontinui.config.state_loader import load_states_from_config
        from qontinui.config.transition_loader import load_transitions_from_config
        from qontinui.model.state.state_service import StateService
        from qontinui.multistate_integration.pathfinding_navigator import (
            PathfindingNavigator,
        )
        from qontinui.state_management.state_memory import StateMemory

        # Step 1: Create StateService
        _state_service = StateService()
        _debug_print("StateService created")
        logger.debug("StateService created")

        # Step 2: Load states from config
        _debug_print("Loading states from config...")
        if not load_states_from_config(config_dict, _state_service):
            _debug_print("ERROR: Failed to load states from configuration")
            logger.error("Failed to load states from configuration")
            return False

        state_count = len(_state_service.get_all_states())
        _debug_print(f"Loaded {state_count} states")
        logger.info(f"Loaded {state_count} states from configuration")

        # Step 3: Load transitions from config
        _debug_print("Step 3: Loading transitions from config...")
        transitions_loaded = load_transitions_from_config(config_dict, _state_service)
        _debug_print(f"load_transitions_from_config returned: {transitions_loaded}")
        if not transitions_loaded:
            _debug_print("ERROR: Failed to load transitions from configuration!")
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

        _debug_print(f"Registered {registered_count} states with multistate adapter")

        # Step 8: Register all transitions with the multistate adapter
        transition_count = 0
        states_with_transitions = 0
        _debug_print(f"Checking transitions on {len(_state_service.get_all_states())} states")
        for state in _state_service.get_all_states():
            _debug_print(
                f"  State '{state.name}' (id={state.id}): {len(state.transitions)} transitions"
            )
            if state.transitions:
                states_with_transitions += 1
                _debug_print(f"    State '{state.name}' has {len(state.transitions)} transitions")
            for transition in state.transitions:
                _debug_print(
                    f"    Registering transition: {transition.id} from state '{state.name}'"  # type: ignore[attr-defined]
                )
                _navigator.multistate_adapter.register_qontinui_transition(transition)  # type: ignore[arg-type]
                transition_count += 1

        _debug_print(f"Found {states_with_transitions} states with transitions")
        _debug_print(f"Registered {transition_count} total transitions with multistate adapter")

        _initialized = True
        _debug_print(f"Navigation system initialized successfully! _initialized={_initialized}")

        logger.info(f"Navigation system initialized successfully with {state_count} states")
        return True

    except Exception as e:
        _debug_print(f"EXCEPTION in load_configuration: {e}")
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        _initialized = False
        return False


def _find_active_states_visually(states: list[Any]) -> set[int]:
    """Find which states are currently visible on screen by searching for their StateImages.

    Collects all StateImages from all target states and performs a parallel
    find_async operation, then maps found patterns back to their owning states.

    This uses async parallel pattern matching for efficient multi-StateImage search.

    Args:
        states: List of State objects to check

    Returns:
        Set of state IDs that are currently active (have visible StateImages)
    """
    import asyncio

    from qontinui.actions.find import FindAction
    from qontinui.actions.find.find_options_builder import CascadeContext, build_find_options
    from qontinui.model.element import Pattern

    active_state_ids: set[int] = set()

    # Collect all StateImages with their owning state IDs
    # Structure: list of (state_id, state_name, state_image, pattern)
    all_state_images: list[tuple[int, str, Any, Pattern]] = []

    for state in states:
        state_id = state.id
        if state_id is None:
            continue

        _debug_print(f"Collecting StateImages for state '{state.name}' (ID: {state_id})")
        _debug_print(f"  State has {len(state.state_images)} StateImages")

        for idx, state_image in enumerate(state.state_images):
            image_name = state_image.name or f"image_{idx}"
            try:
                pattern = state_image.get_pattern()
                all_state_images.append((state_id, state.name, state_image, pattern))
                _debug_print(f"  Added StateImage '{image_name}'")
            except Exception as e:
                _debug_print(f"  Error getting pattern for '{image_name}': {e}")

    if not all_state_images:
        _debug_print("No StateImages found for any target states")
        logger.warning("FIND_STATE check: No StateImages found for target states")
        return active_state_ids

    _debug_print(
        f"Performing find_async on {len(all_state_images)} StateImages from {len(states)} states..."
    )

    # Get project config for cascade
    try:
        from qontinui.config.settings import QontinuiSettings

        project_config = QontinuiSettings()
    except Exception:
        project_config = None

    find_action = FindAction()

    # Extract patterns for parallel find
    patterns = [item[3] for item in all_state_images]

    # Build find options (using first state image for cascade context)
    ctx = CascadeContext(
        search_options=None,
        pattern=patterns[0] if patterns else None,
        state_image=all_state_images[0][2] if all_state_images else None,
        project_config=project_config,
        monitor_index=None,
    )
    options = build_find_options(ctx)

    try:
        # Run async find in a new event loop (since we're in sync context)
        async def run_find_async() -> list[Any]:
            return await find_action.find(patterns, options)

        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, need to use run_coroutine_threadsafe

            future = asyncio.run_coroutine_threadsafe(run_find_async(), loop)
            results = future.result(timeout=30.0)
        except RuntimeError:
            # No running event loop, create one
            results = asyncio.run(run_find_async())

        # Map results back to states
        for i, result in enumerate(results):
            state_id, state_name, state_image, pattern = all_state_images[i]
            image_name = state_image.name or f"image_{i}"

            if result.found:
                _debug_print(f"  FOUND! State '{state_name}' is active (StateImage '{image_name}')")
                logger.info(f"State '{state_name}' is active (found StateImage '{image_name}')")
                active_state_ids.add(state_id)
            else:
                _debug_print(f"  StateImage '{image_name}' not found on screen")

    except Exception as e:
        _debug_print(f"Error in find: {e}")
        logger.warning(f"Error in parallel find operation: {e}")

    _debug_print(f"FIND_STATE check complete: {len(active_state_ids)}/{len(states)} states active")
    logger.info(f"FIND_STATE check: {len(active_state_ids)}/{len(states)} states visible on screen")
    return active_state_ids


def open_state(state_name: str) -> bool:
    """Navigate to the specified state by name.

    This is a convenience wrapper around open_states() for single state navigation.

    Args:
        state_name: Name of the state to navigate to

    Returns:
        True if navigation succeeded, False otherwise
    """
    return open_states([state_name])


def open_states(state_identifiers: list[str | int]) -> bool:
    """Navigate to multiple specified states.

    First performs a FIND_STATE check to see if all target states are already
    visible on screen. If so, returns success without navigation.

    This function finds a path that activates ALL specified target states.
    The library handles all pathfinding, state detection, and transition execution.

    Args:
        state_identifiers: List of state names (str) or state IDs (int) to navigate to.
                          All states must be reached for the navigation to succeed.

    Returns:
        True if navigation succeeded and ALL target states were reached, False otherwise
    """
    _debug_print(f"open_states() called with: {state_identifiers}")
    _debug_print(
        f"_initialized={_initialized}, _navigator={_navigator is not None}, _state_service={_state_service is not None}"
    )

    if not _initialized:
        _debug_print("ERROR: Navigation system not initialized!")
        logger.error("Navigation system not initialized - call load_configuration() first")
        return False

    if not _navigator:
        _debug_print("ERROR: Navigator not available!")
        logger.error("Navigator not available")
        return False

    if not _state_service:
        _debug_print("ERROR: State service not available!")
        logger.error(
            "State service not available - state/transition loading from config not yet implemented"
        )
        return False

    if not state_identifiers:
        _debug_print("ERROR: No state identifiers provided!")
        logger.error("No state identifiers provided")
        return False

    # Convert all identifiers to state IDs and collect State objects
    target_state_ids = []
    target_states = []
    for identifier in state_identifiers:
        if isinstance(identifier, int):
            state_id = identifier
            state = _state_service.get_state_by_id(state_id)  # type: ignore[attr-defined]
            if not state:
                _debug_print(f"ERROR: State ID {state_id} not found!")
                logger.error(f"State ID {state_id} not found in state service")
                return False
        elif isinstance(identifier, str):
            state = _state_service.get_state_by_name(identifier)
            if not state or not state.id:
                _debug_print(f"ERROR: State '{identifier}' not found!")
                logger.error(f"State '{identifier}' not found in state service")
                return False
            state_id = state.id
            _debug_print(f"State '{identifier}' -> ID {state_id}")
        else:
            _debug_print(f"ERROR: Invalid state identifier type: {type(identifier)}")
            logger.error(f"Invalid state identifier type: {type(identifier)}")
            return False

        target_state_ids.append(state_id)
        target_states.append(state)

    _debug_print(f"Target state IDs: {target_state_ids}")
    logger.info(f"Navigating to states: {state_identifiers} (IDs: {target_state_ids})")

    # FIND_STATE check: Visually verify if all target states are already active
    _debug_print("Performing FIND_STATE check for target states...")
    logger.info("Performing FIND_STATE check for target states")
    active_states_found = _find_active_states_visually(target_states)

    if active_states_found and all(s.id in active_states_found for s in target_states):
        state_names = [s.name for s in target_states]
        _debug_print(f"All target states already visible on screen: {state_names}")
        logger.info(
            f"All target states already active on screen: {state_names} - skipping navigation"
        )
        return True

    # Log which states are missing
    missing_states = [s for s in target_states if s.id not in active_states_found]
    missing_names = [s.name for s in missing_states]
    _debug_print(f"States not yet visible: {missing_names}. Proceeding with navigation.")
    logger.info(f"States not yet active: {missing_names}. Initiating navigation.")

    # Get current active states for debugging
    active_states = _state_memory.get_active_state_names() if _state_memory else []
    active_state_ids = list(_state_memory.active_states) if _state_memory else []
    _debug_print(f"Current active states (from memory): {active_states} (IDs: {active_state_ids})")
    logger.info(f"Current active states (from memory): {active_states}")

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
