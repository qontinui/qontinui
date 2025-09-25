"""OutgoingTransition annotation for Qontinui framework.

Marks methods as transitions FROM a specific state (outgoing transitions).
Ported from Brobot's transition system.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any


def outgoing_transition(
    from_state: type, priority: int = 0, description: str = "", timeout: int = 10
) -> Any:
    """Marks a method as an OutgoingTransition.

    An OutgoingTransition is a transition FROM a specific state (outgoing from that state)
    TO the state(s) defined in the enclosing @transition_set class.

    The annotated method should:
    - Return boolean (true if transition succeeds, false otherwise)
    - Contain the actions needed to navigate FROM the source state
    - Be a member of a class annotated with @transition_set

    Example usage:
        @outgoing_transition(from_state=MenuState, priority=1)
        def from_menu(self) -> bool:
            logger.info("Navigating from Menu (outgoing transition)")
            return self.action.click(self.menu_state.pricing_button).is_success()

    Args:
        from_state: The source state class for this transition.
                   This transition will navigate FROM this state (outgoing)
                   TO the state(s) defined in @transition_set.
        priority: Priority of this transition when multiple paths exist.
                 Higher values indicate higher priority. Default is 0.
        description: Optional description of this transition.
                    Useful for documentation and debugging.
        timeout: Timeout for this transition in seconds. Default is 10 seconds.

    Returns:
        The decorated method with transition metadata attached.
    """

    def decorator(method: Callable) -> Callable:
        # Store metadata on the method
        method._qontinui_outgoing_transition = True
        method._qontinui_outgoing_transition_from = from_state
        method._qontinui_outgoing_transition_priority = priority
        method._qontinui_outgoing_transition_description = description
        method._qontinui_outgoing_transition_timeout = timeout

        @wraps(method)
        def wrapper(*args, **kwargs):
            return method(*args, **kwargs)

        # Copy metadata to wrapper
        wrapper._qontinui_outgoing_transition = True
        wrapper._qontinui_outgoing_transition_from = from_state
        wrapper._qontinui_outgoing_transition_priority = priority
        wrapper._qontinui_outgoing_transition_description = description
        wrapper._qontinui_outgoing_transition_timeout = timeout

        return wrapper

    return decorator


def is_outgoing_transition(method: Any) -> bool:
    """Check if a method is an OutgoingTransition.

    Args:
        method: Method to check

    Returns:
        True if method is decorated with @outgoing_transition
    """
    return hasattr(method, "_qontinui_outgoing_transition") and method._qontinui_outgoing_transition


def get_outgoing_transition_metadata(method: Any) -> dict | None:
    """Get OutgoingTransition metadata from a decorated method.

    Args:
        method: The transition method

    Returns:
        Dictionary with transition metadata or None
    """
    if not is_outgoing_transition(method):
        return None

    return {
        "from_state": getattr(method, "_qontinui_outgoing_transition_from", None),
        "priority": getattr(method, "_qontinui_outgoing_transition_priority", 0),
        "description": getattr(method, "_qontinui_outgoing_transition_description", ""),
        "timeout": getattr(method, "_qontinui_outgoing_transition_timeout", 10),
    }
