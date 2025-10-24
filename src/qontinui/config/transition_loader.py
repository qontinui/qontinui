"""Transition Loader - loads transitions from configuration into StateService.

This module implements Phase 3 of the State/Transition Loading Implementation Plan.
It parses transition definitions from JSON configuration and creates StateTransition
objects that are linked to their source and target states.

The loader handles:
- Creating TaskSequenceStateTransition objects from config
- Looking up states by string IDs
- Linking transitions to workflows from the registry
- Setting transition properties (timeout, retryCount, etc.)
- Managing multi-state activation/deactivation
- Error handling for missing states or workflows
"""

from __future__ import annotations

import logging
from typing import Any

from qontinui import registry
from qontinui.model.state.state_service import StateService
from qontinui.model.transition.enhanced_state_transition import (
    StaysVisible,
    TaskSequenceStateTransition,
)

logger = logging.getLogger(__name__)


def load_transitions_from_config(config: dict[str, Any], state_service: StateService) -> bool:
    """Load all transitions from configuration and link them to states.

    This function is the main entry point for Phase 3 of the state loading process.
    It extracts transition definitions from the config JSON, creates StateTransition
    objects, links them to workflows, and adds them to their source states.

    Args:
        config: Full configuration dictionary containing "transitions" array
        state_service: StateService with already-loaded states

    Returns:
        True if all transitions loaded successfully, False if any errors occurred

    Example:
        >>> config = {
        ...     "transitions": [
        ...         {
        ...             "id": "trans-1",
        ...             "type": "OutgoingTransition",
        ...             "processes": ["process-demo-1"],
        ...             "timeout": 10000,
        ...             "retryCount": 3,
        ...             "fromState": "state-start",
        ...             "toState": "state-middle",
        ...             "staysVisible": false,
        ...             "activateStates": [],
        ...             "deactivateStates": []
        ...         }
        ...     ]
        ... }
        >>> success = load_transitions_from_config(config, state_service)
    """
    if "transitions" not in config:
        logger.warning("No 'transitions' key found in configuration")
        return True  # Not an error - config may have no transitions

    transitions = config["transitions"]
    if not isinstance(transitions, list):
        logger.error("Configuration 'transitions' must be a list")
        return False

    logger.info(f"Loading {len(transitions)} transitions from configuration")

    success_count = 0
    error_count = 0

    for transition_def in transitions:
        try:
            if _load_single_transition(transition_def, state_service):
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            logger.error(f"Unexpected error loading transition: {e}", exc_info=True)
            error_count += 1

    logger.info(
        f"Transition loading complete: {success_count} succeeded, {error_count} failed"
    )

    # Return True only if all transitions loaded successfully
    return error_count == 0


def _load_single_transition(
    transition_def: dict[str, Any], state_service: StateService
) -> bool:
    """Load a single transition from its definition.

    Args:
        transition_def: Transition definition from config
        state_service: StateService for looking up states

    Returns:
        True if transition loaded successfully, False otherwise
    """
    transition_id = transition_def.get("id", "<unknown>")
    transition_type = transition_def.get("type", "OutgoingTransition")

    # Validate required fields
    if not _validate_transition_definition(transition_def, transition_id, transition_type):
        return False

    # IncomingTransition types don't require a fromState - they represent
    # verification workflows that execute when entering a state
    if transition_type == "IncomingTransition":
        # Look up toState
        to_state_id = transition_def["toState"]
        to_state = state_service.get_state_by_name(to_state_id)
        if to_state is None:
            logger.error(
                f"Transition '{transition_id}': toState '{to_state_id}' not found"
            )
            return False

        # Create the transition object (no fromState for incoming)
        transition = _create_transition_object(
            transition_def, None, to_state.id, state_service
        )
        if transition is None:
            return False

        # Link workflows to the transition
        if not _link_workflows_to_transition(transition_def, transition, transition_id):
            logger.warning(
                f"Transition '{transition_id}': No workflows linked (this may be intentional)"
            )

        # Add to target state's incoming_transitions
        to_state.incoming_transitions.append(transition)
        logger.info(
            f"Loaded IncomingTransition '{transition_id}' to state '{to_state.name}' "
            f"with {len(transition.workflow_ids)} workflows"
        )
        return True

    # OutgoingTransition - requires both fromState and toState
    from_state_id = transition_def["fromState"]
    from_state = state_service.get_state_by_name(from_state_id)
    if from_state is None:
        logger.error(
            f"Transition '{transition_id}': fromState '{from_state_id}' not found"
        )
        return False

    # Look up toState
    to_state_id = transition_def["toState"]
    to_state = state_service.get_state_by_name(to_state_id)
    if to_state is None:
        logger.error(
            f"Transition '{transition_id}': toState '{to_state_id}' not found"
        )
        return False

    # Create the transition object
    transition = _create_transition_object(
        transition_def, from_state.id, to_state.id, state_service
    )
    if transition is None:
        return False

    # Link workflows to the transition
    if not _link_workflows_to_transition(transition_def, transition, transition_id):
        # Log warning but don't fail - transition may not need workflows
        logger.warning(
            f"Transition '{transition_id}': No workflows linked (this may be intentional)"
        )

    # Add transition to the source state
    from_state.add_transition(transition)

    # Debug print to bypass logging
    print(f"DEBUG: ADDED transition '{transition_id}' to state '{from_state.name}' - state now has {len(from_state.transitions)} transitions", flush=True)

    return True


def _validate_transition_definition(
    transition_def: dict[str, Any], transition_id: str, transition_type: str
) -> bool:
    """Validate that a transition definition has all required fields.

    Args:
        transition_def: Transition definition to validate
        transition_id: ID of transition (for error messages)
        transition_type: Type of transition (OutgoingTransition or IncomingTransition)

    Returns:
        True if valid, False otherwise
    """
    # All transitions require a toState
    if "toState" not in transition_def:
        logger.error(
            f"Transition '{transition_id}': missing required field 'toState'"
        )
        return False

    # OutgoingTransitions require a fromState
    # IncomingTransitions do not (they represent arrival from any source)
    if transition_type == "OutgoingTransition" and "fromState" not in transition_def:
        logger.error(
            f"Transition '{transition_id}': OutgoingTransition missing required field 'fromState'"
        )
        return False

    return True


def _extract_numeric_id(transition_id: str) -> int:
    """Extract or generate a numeric ID from a string transition ID.

    Args:
        transition_id: String ID from config (e.g., "transition-1759519237735")

    Returns:
        Integer ID suitable for use in TaskSequenceStateTransition
    """
    # Try to extract the last numeric part from the ID
    # Format is typically "transition-<timestamp>" or similar
    parts = transition_id.split("-")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)

    # If no numeric part found, use hash of the string
    # Use abs() to ensure positive integer
    return abs(hash(transition_id))


def _create_transition_object(
    transition_def: dict[str, Any],
    from_state_int_id: int | None,
    to_state_int_id: int | None,
    state_service: StateService,
) -> TaskSequenceStateTransition | None:
    """Create a TaskSequenceStateTransition object from config.

    Args:
        transition_def: Transition definition from config
        from_state_int_id: Integer ID of source state
        to_state_int_id: Integer ID of target state
        state_service: StateService for looking up additional states

    Returns:
        TaskSequenceStateTransition object, or None if creation failed
    """
    transition_id = transition_def.get("id", "<unknown>")

    try:
        # Extract numeric ID from string ID (e.g., "transition-1759519237735" -> 1759519237735)
        # If extraction fails, use hash of the string ID
        numeric_id = _extract_numeric_id(transition_id)

        # Create the transition with basic properties including the ID
        transition = TaskSequenceStateTransition(
            id=numeric_id,
            name=transition_def.get("type", "Transition"),
            description=f"Transition {transition_id}",
            path_cost=1,  # Default cost
        )

        # Set source states
        if from_state_int_id is not None:
            transition.from_states.add(from_state_int_id)

        # Set target states (activate)
        if to_state_int_id is not None:
            transition.activate.add(to_state_int_id)

        # Parse stays_visible
        stays_visible_value = transition_def.get("staysVisible", False)
        if isinstance(stays_visible_value, bool):
            transition.stays_visible = (
                StaysVisible.TRUE if stays_visible_value else StaysVisible.FALSE
            )
        elif isinstance(stays_visible_value, str):
            try:
                transition.stays_visible = StaysVisible[stays_visible_value.upper()]
            except KeyError:
                logger.warning(
                    f"Transition '{transition_id}': invalid staysVisible value "
                    f"'{stays_visible_value}', using FALSE"
                )
                transition.stays_visible = StaysVisible.FALSE

        # Parse activateStates (additional states to activate)
        activate_states = transition_def.get("activateStates", [])
        if isinstance(activate_states, list):
            for state_name in activate_states:
                state = state_service.get_state_by_name(state_name)
                if state and state.id is not None:
                    transition.activate.add(state.id)
                else:
                    logger.warning(
                        f"Transition '{transition_id}': activateState '{state_name}' not found"
                    )

        # Parse deactivateStates (states to exit)
        deactivate_states = transition_def.get("deactivateStates", [])
        if isinstance(deactivate_states, list):
            for state_name in deactivate_states:
                state = state_service.get_state_by_name(state_name)
                if state and state.id is not None:
                    transition.exit.add(state.id)
                else:
                    logger.warning(
                        f"Transition '{transition_id}': deactivateState '{state_name}' not found"
                    )

        # Optional: parse timeout and retryCount (not currently used by TaskSequenceStateTransition)
        # These could be stored in metadata or used by execution layer
        timeout = transition_def.get("timeout")
        retry_count = transition_def.get("retryCount")
        if timeout is not None or retry_count is not None:
            logger.debug(
                f"Transition '{transition_id}': timeout={timeout}, retryCount={retry_count} "
                "(not currently stored in transition object)"
            )

        return transition

    except Exception as e:
        logger.error(
            f"Transition '{transition_id}': failed to create transition object: {e}",
            exc_info=True,
        )
        return None


def _link_workflows_to_transition(
    transition_def: dict[str, Any],
    transition: TaskSequenceStateTransition,
    transition_id: str,
) -> bool:
    """Link workflows/processes to a transition.

    Args:
        transition_def: Transition definition from config
        transition: Transition object to link workflows to
        transition_id: ID of transition (for error messages)

    Returns:
        True if at least one workflow was linked, False otherwise
    """
    workflows = transition_def.get("workflows", [])
    if not isinstance(workflows, list):
        logger.warning(
            f"Transition '{transition_id}': 'workflows' field is not a list"
        )
        return False

    if len(workflows) == 0:
        # No workflows specified - this may be intentional (e.g., state just appears)
        return False

    linked_count = 0

    for workflow_id in workflows:
        workflow = registry.get_workflow(workflow_id)
        if workflow is None:
            logger.error(
                f"Transition '{transition_id}': workflow '{workflow_id}' not found in registry"
            )
            continue

        # Store workflow ID in transition for later execution
        transition.workflow_ids.append(workflow_id)
        logger.debug(
            f"Transition '{transition_id}': linked to workflow '{workflow_id}'"
        )
        linked_count += 1

    return linked_count > 0


# Additional helper functions for future expansion


def get_transition_statistics(state_service: StateService) -> dict[str, Any]:
    """Get statistics about loaded transitions.

    Args:
        state_service: StateService to analyze

    Returns:
        Dictionary with transition statistics
    """
    total_transitions = 0
    states_with_transitions = 0
    max_transitions_per_state = 0

    for state in state_service.get_all_states():
        transition_count = len(state.transitions)
        total_transitions += transition_count

        if transition_count > 0:
            states_with_transitions += 1
            max_transitions_per_state = max(max_transitions_per_state, transition_count)

    return {
        "total_transitions": total_transitions,
        "total_states": len(state_service.get_all_states()),
        "states_with_transitions": states_with_transitions,
        "states_without_transitions": len(state_service.get_all_states())
        - states_with_transitions,
        "max_transitions_per_state": max_transitions_per_state,
        "avg_transitions_per_state": (
            total_transitions / len(state_service.get_all_states())
            if len(state_service.get_all_states()) > 0
            else 0
        ),
    }


def validate_transition_graph(state_service: StateService) -> list[str]:
    """Validate the loaded transition graph for common issues.

    Args:
        state_service: StateService to validate

    Returns:
        List of warning/error messages (empty if no issues found)
    """
    issues = []

    # Check for states with no outgoing transitions (except intended final states)
    states_without_outgoing = []
    for state in state_service.get_all_states():
        if len(state.transitions) == 0:
            states_without_outgoing.append(state.name)

    if states_without_outgoing:
        issues.append(
            f"States with no outgoing transitions: {', '.join(states_without_outgoing)}"
        )

    # Check for unreachable states (states with no incoming transitions)
    all_target_states = set()
    for state in state_service.get_all_states():
        for transition in state.transitions:
            all_target_states.update(transition.activate)

    unreachable_states = []
    for state in state_service.get_all_states():
        if state.id is not None and state.id not in all_target_states:
            # This state is not the target of any transition
            unreachable_states.append(state.name)

    if unreachable_states:
        issues.append(
            f"Potentially unreachable states: {', '.join(unreachable_states)} "
            "(no incoming transitions)"
        )

    return issues
