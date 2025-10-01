"""State annotation - ported from Qontinui framework.

Marks classes as Qontinui states for automatic registration.
"""

from typing import Any


def state(initial: bool = False, name: str = "", description: str = "") -> Any:
    """Annotation for Qontinui states.

    Direct port of Brobot's @State annotation.

    This decorator marks a class as a Qontinui state and enables
    automatic registration with the state management system.

    Classes decorated with @state should include:
    - StateObject attributes for UI elements
    - Methods for state-specific operations

    Usage:
        @state
        class PromptState:
            def __init__(self):
                self.submit_button = StateObject.builder()\\
                    .with_image("submit")\\
                    .build()

    To mark as initial state:
        @state(initial=True)
        class InitialState:
            # state definition

    Args:
        initial: Whether this state is an initial state.
                Initial states are automatically registered as
                starting points for the state machine.
        name: Optional name for the state. If not specified,
             the class name (without "State" suffix if present)
             will be used.
        description: Optional description of the state's purpose.
                    Used for documentation and debugging.

    Returns:
        The decorated class with state metadata attached.
    """

    def decorator(cls: type) -> type:
        # Store metadata on the class
        cls._qontinui_state = True  # type: ignore[attr-defined]
        cls._qontinui_state_initial = initial  # type: ignore[attr-defined]
        cls._qontinui_state_name = name or _derive_state_name(cls)  # type: ignore[attr-defined]
        cls._qontinui_state_description = description  # type: ignore[attr-defined]

        # Register with state registry if available
        # This will be handled by StateAnnotationProcessor

        return cls

    return decorator


def _derive_state_name(cls: type) -> str:
    """Derive state name from class name.

    Args:
        cls: The state class

    Returns:
        Derived state name
    """
    class_name = cls.__name__

    # Remove "State" suffix if present
    if class_name.endswith("State"):
        return class_name[:-5]

    return class_name


def is_state(obj: Any) -> bool:
    """Check if an object is a Qontinui state.

    Args:
        obj: Object to check

    Returns:
        True if object is decorated with @state
    """
    return hasattr(obj, "_qontinui_state") and obj._qontinui_state


def get_state_metadata(cls: type) -> dict[str, Any] | None:
    """Get state metadata from a decorated class.

    Args:
        cls: The state class

    Returns:
        Dictionary with state metadata or None
    """
    if not is_state(cls):
        return None

    return {
        "initial": getattr(cls, "_qontinui_state_initial", False),
        "name": getattr(cls, "_qontinui_state_name", ""),
        "description": getattr(cls, "_qontinui_state_description", ""),
    }
