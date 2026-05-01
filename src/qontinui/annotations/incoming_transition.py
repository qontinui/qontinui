"""IncomingTransition annotation for Qontinui framework.

Marks methods as IncomingTransitions (arrival verification transitions).
Ported from Brobot's transition system.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any


def incoming_transition(
    description: str = "", timeout: int = 5, required: bool = True
) -> Any:
    """Marks a method as an IncomingTransition.

    An IncomingTransition (arrival/finish transition) verifies that we have
    successfully arrived at the target state (incoming to that state).

    The annotated method should:
    - Return boolean (true if state is confirmed, false otherwise)
    - Verify the presence of unique elements that confirm the state is active
    - Be a member of a class annotated with @transition_set
    - There should be only ONE @incoming_transition method per @transition_set class

    This transition is executed after any OutgoingTransition to confirm successful
    navigation to the target state, regardless of which state we came from.

    Example usage:
        @incoming_transition
        def verify_arrival(self) -> bool:
            logger.info("Verifying arrival at Pricing state (incoming transition)")
            return self.action.find(self.pricing_state.start_for_free_button).is_success()

    Args:
        description: Optional description of this arrival verification.
                    Useful for documentation and debugging.
        timeout: Timeout for verifying arrival in seconds. Default is 5 seconds.
        required: Whether this verification is required for the transition
                 to be considered successful. If false, failure of this
                 verification will log a warning but not fail the transition.
                 Default is True.

    Returns:
        The decorated method with transition metadata attached.
    """

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        # Store metadata on the method
        method._qontinui_incoming_transition = True  # type: ignore[attr-defined]
        method._qontinui_incoming_transition_description = description  # type: ignore[attr-defined]
        method._qontinui_incoming_transition_timeout = timeout  # type: ignore[attr-defined]
        method._qontinui_incoming_transition_required = required  # type: ignore[attr-defined]

        @wraps(method)
        def wrapper(*args, **kwargs):
            return method(*args, **kwargs)

        # Copy metadata to wrapper
        wrapper._qontinui_incoming_transition = True  # type: ignore[attr-defined]
        wrapper._qontinui_incoming_transition_description = description  # type: ignore[attr-defined]
        wrapper._qontinui_incoming_transition_timeout = timeout  # type: ignore[attr-defined]
        wrapper._qontinui_incoming_transition_required = required  # type: ignore[attr-defined]

        return wrapper

    return decorator


def is_incoming_transition(method: Any) -> bool:
    """Check if a method is an IncomingTransition.

    Args:
        method: Method to check

    Returns:
        True if method is decorated with @incoming_transition
    """
    return (
        hasattr(method, "_qontinui_incoming_transition")
        and method._qontinui_incoming_transition
    )


def get_incoming_transition_metadata(method: Any) -> dict[str, Any] | None:
    """Get IncomingTransition metadata from a decorated method.

    Args:
        method: The transition method

    Returns:
        Dictionary with transition metadata or None
    """
    if not is_incoming_transition(method):
        return None

    return {
        "description": getattr(method, "_qontinui_incoming_transition_description", ""),
        "timeout": getattr(method, "_qontinui_incoming_transition_timeout", 5),
        "required": getattr(method, "_qontinui_incoming_transition_required", True),
    }
