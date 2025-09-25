"""Transition method annotation - ported from Brobot.

Marks methods as state transitions for automatic registration.
Following Brobot's @Transition annotation pattern for methods.
"""

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


def transition(
    from_state: str | None = None,
    to_state: str | None = None,
    activate: str | set[str] | list[str] | None = None,
    exit: str | set[str] | list[str] | None = None,
    name: str | None = None,
    description: str = "",
    priority: int = 50,
) -> Any:
    """Method decorator for state transitions.

    Direct port of Brobot's @Transition annotation for methods.

    This decorator marks a method as a state transition and enables
    automatic registration with the state management system.

    Methods decorated with @transition should:
    - Return bool indicating success/failure
    - Perform the actions needed to transition between states
    - Be idempotent when possible

    Usage in a state class:
        @state
        class LoginState:
            def __init__(self, actions):
                self.actions = actions

            @transition(to_state="Dashboard")
            def login_to_dashboard(self) -> bool:
                # Perform login actions
                return self.actions.click(submit_button).success

            @transition(to_state="ForgotPassword", priority=30)
            def go_to_forgot_password(self) -> bool:
                return self.actions.click(forgot_link).success

    Usage in a transitions class:
        class AppTransitions:
            def __init__(self, actions):
                self.actions = actions

            @transition(
                from_state="Dashboard",
                to_state="Settings",
                activate={"Settings", "SettingsMenu"},
                exit={"Dashboard"}
            )
            def open_settings(self) -> bool:
                # Open settings
                return self.actions.click(settings_icon).success

            @transition(
                from_state="Settings",
                to_state="Dashboard",
                description="Close settings and return to dashboard"
            )
            def close_settings(self) -> bool:
                return self.actions.key("ESC").success

    Args:
        from_state: Name of the source state. If None, uses the containing
                   class name if it's a @state decorated class.
        to_state: Name of the target state.
        activate: State name(s) to activate on success. If None and to_state
                 is specified, defaults to {to_state}.
        exit: State name(s) to deactivate on success. If None and from_state
             is specified, defaults to {from_state}.
        name: Optional name for the transition. If not specified,
              the method name will be used.
        description: Optional description of the transition's purpose.
        priority: Priority for transition selection (0-100, higher = preferred).
                 Default is 50.

    Returns:
        The decorated method with transition metadata attached.
    """

    def decorator(func: Callable) -> Callable:
        # Normalize activate/exit to sets
        if activate is None:
            activate_set = {to_state} if to_state else set()
        elif isinstance(activate, str):
            activate_set = {activate}
        elif isinstance(activate, list | tuple):
            activate_set = set(activate)
        else:
            activate_set = activate

        if exit is None:
            exit_set = {from_state} if from_state else set()
        elif isinstance(exit, str):
            exit_set = {exit}
        elif isinstance(exit, list | tuple):
            exit_set = set(exit)
        else:
            exit_set = exit

        # Store metadata on the function
        func._qontinui_transition = True
        func._qontinui_transition_from = from_state
        func._qontinui_transition_to = to_state
        func._qontinui_transition_activate = activate_set
        func._qontinui_transition_exit = exit_set
        func._qontinui_transition_name = name or func.__name__
        func._qontinui_transition_description = description
        func._qontinui_transition_priority = priority

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Log transition attempt
            logger.debug(f"Executing transition: {func._qontinui_transition_name}")

            # Execute the transition
            result = func(*args, **kwargs)

            # Log result
            if result:
                logger.info(f"Transition succeeded: {func._qontinui_transition_name}")
            else:
                logger.warning(f"Transition failed: {func._qontinui_transition_name}")

            return result

        # Copy metadata to wrapper
        wrapper._qontinui_transition = func._qontinui_transition
        wrapper._qontinui_transition_from = func._qontinui_transition_from
        wrapper._qontinui_transition_to = func._qontinui_transition_to
        wrapper._qontinui_transition_activate = func._qontinui_transition_activate
        wrapper._qontinui_transition_exit = func._qontinui_transition_exit
        wrapper._qontinui_transition_name = func._qontinui_transition_name
        wrapper._qontinui_transition_description = func._qontinui_transition_description
        wrapper._qontinui_transition_priority = func._qontinui_transition_priority

        return wrapper

    return decorator


def is_transition_method(obj: Any) -> bool:
    """Check if an object is a transition method.

    Args:
        obj: Object to check

    Returns:
        True if object is decorated with @transition
    """
    return hasattr(obj, "_qontinui_transition") and obj._qontinui_transition


def get_transition_metadata(func: Callable) -> dict | None:
    """Get transition metadata from a decorated method.

    Args:
        func: The transition method

    Returns:
        Dictionary with transition metadata or None
    """
    if not is_transition_method(func):
        return None

    return {
        "from_state": getattr(func, "_qontinui_transition_from", None),
        "to_state": getattr(func, "_qontinui_transition_to", None),
        "activate": getattr(func, "_qontinui_transition_activate", set()),
        "exit": getattr(func, "_qontinui_transition_exit", set()),
        "name": getattr(func, "_qontinui_transition_name", ""),
        "description": getattr(func, "_qontinui_transition_description", ""),
        "priority": getattr(func, "_qontinui_transition_priority", 50),
        "function": func,
    }


def collect_transitions(obj: Any) -> list[dict]:
    """Collect all transition methods from an object.

    Scans an object (typically a class instance) for methods
    decorated with @transition and returns their metadata.

    Args:
        obj: Object to scan for transitions

    Returns:
        List of transition metadata dictionaries
    """
    transitions = []

    # Get the class name if it's a @state decorated class
    from .state import get_state_metadata, is_state

    default_from_state = None
    if is_state(obj.__class__):
        state_meta = get_state_metadata(obj.__class__)
        if state_meta:
            default_from_state = state_meta["name"]

    # Scan all attributes
    for attr_name in dir(obj):
        if attr_name.startswith("_"):
            continue

        try:
            attr = getattr(obj, attr_name)
            if is_transition_method(attr):
                metadata = get_transition_metadata(attr)
                if metadata:
                    # Use class state name as from_state if not specified
                    if metadata["from_state"] is None and default_from_state:
                        metadata["from_state"] = default_from_state
                    transitions.append(metadata)
        except AttributeError:
            continue

    # Sort by priority (higher first)
    transitions.sort(key=lambda x: x["priority"], reverse=True)

    return transitions


def create_code_transition(metadata: dict):
    """Create a CodeStateTransition from transition metadata.

    Args:
        metadata: Transition metadata dictionary

    Returns:
        CodeStateTransition instance
    """
    from ..navigation.transition.code_state_transition import CodeStateTransition

    return CodeStateTransition(
        name=metadata["name"],
        from_state=metadata["from_state"],
        to_state=metadata["to_state"],
        activate_names=metadata["activate"],
        exit_names=metadata["exit"],
        transition_function=metadata["function"],
    )
