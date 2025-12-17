"""Enhanced state decorator with full Brobot features.

This module provides decorators for defining states and transitions with:
- Priority-based initial state selection
- Profile support for environment-specific configurations
- Path cost integration for pathfinding
- State group management
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StateMetadata:
    """Metadata for a state class."""

    name: str
    initial: bool = False
    priority: int = 100
    path_cost: int = 1
    profiles: list[str] = field(default_factory=list)
    group: str | None = None
    description: str = ""
    blocking: bool = False
    can_hide: list[str] = field(default_factory=list)


def state(
    initial: bool = False,
    name: str = "",
    description: str = "",
    priority: int = 100,
    path_cost: int = 1,
    profiles: list[str] | None = None,
    group: str | None = None,
    blocking: bool = False,
    can_hide: list[str] | None = None,
) -> Callable[[type], type]:
    """Enhanced state decorator with full Brobot features.

    This decorator marks a class as a Qontinui state and enables
    automatic registration with the state management system.

    Args:
        initial: Whether this state is an initial state.
                Initial states are automatically registered as
                starting points for the state machine.
        name: Optional name for the state. If not specified,
             the class name (without "State" suffix if present)
             will be used.
        description: Optional description of the state's purpose.
                    Used for documentation and debugging.
        priority: Priority for initial state selection (higher values = higher priority).
                 Used when multiple initial states are defined to influence
                 selection probability. Default is 100 for equal probability
                 among all initial states. Only applies when initial = True.
        path_cost: Path-finding cost for reaching this state.
                  The total cost of a path is the sum of all state costs
                  and transition costs in that path. Lower costs are preferred
                  when multiple paths exist. Default is 1.
                  Example values:
                  - 0: Free state (no cost to be in this state)
                  - 1: Normal state (default)
                  - 5: Slightly expensive state (e.g., requires loading)
                  - 10+: Expensive state to reach (e.g., error recovery states)
        profiles: List of profiles where this state should be considered initial.
                 Empty list means the state is initial in all profiles.
                 Only applies when initial = True.
                 Example: ["test", "development"]
        group: Optional state group name. States in the same group
              often activate together (e.g., "workspace", "dialog").
        blocking: Whether this state blocks access to other states until resolved.
                 Blocking states (like modal dialogs) must be handled before
                 other states can be accessed.
        can_hide: List of state names that this state can hide when active.
                 Hidden states remain in memory but aren't visible.

    Returns:
        The decorated class with state metadata attached.

    Example:
        @state(initial=True, priority=150, group="main_window")
        class MainMenuState:
            def __init__(self) -> None:
                self.file_menu = StateObject.builder()\\
                    .with_image("file_menu")\\
                    .build()

        @state(blocking=True, can_hide=["MainMenuState"])
        class ModalDialogState:
            # This modal blocks other states and hides the main menu
            pass

        @state(profiles=["test"], path_cost=5)
        class TestOnlyState:
            # Only available in test profile, expensive to reach
            pass
    """

    def decorator(cls: type) -> type:
        # Derive state name if not provided
        state_name = name or _derive_state_name(cls)

        # Create metadata
        metadata = StateMetadata(
            name=state_name,
            initial=initial,
            priority=priority,
            path_cost=path_cost,
            profiles=profiles or [],
            group=group,
            description=description or cls.__doc__ or "",
            blocking=blocking,
            can_hide=can_hide or [],
        )

        # Store metadata on the class
        cls._qontinui_state = True  # type: ignore[attr-defined]
        cls._qontinui_state_metadata = metadata  # type: ignore[attr-defined]

        # Log state registration
        logger.debug(
            f"State decorated: {state_name} "
            f"(initial={initial}, priority={priority}, group={group})"
        )

        # Add helper methods to the class
        _add_state_methods(cls, metadata)

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


def _add_state_methods(cls: type, metadata: StateMetadata) -> None:
    """Add helper methods to the state class.

    Args:
        cls: The state class
        metadata: State metadata
    """

    def get_state_name(self) -> str:
        """Get the state name."""
        return metadata.name

    def get_state_metadata(self) -> StateMetadata:
        """Get the state metadata."""
        return metadata

    def is_initial_state(self, profile: str | None = None) -> bool:
        """Check if this is an initial state for the given profile."""
        if not metadata.initial:
            return False

        if not profile or not metadata.profiles:
            return True

        return profile in metadata.profiles

    def get_path_cost(self) -> int:
        """Get the path cost for this state."""
        return metadata.path_cost

    def is_blocking(self) -> bool:
        """Check if this is a blocking state."""
        return metadata.blocking

    def can_hide_state(self, state_name: str) -> bool:
        """Check if this state can hide another state."""
        return state_name in metadata.can_hide

    # Add methods to the class
    cls.get_state_name = get_state_name  # type: ignore[attr-defined]
    cls.get_state_metadata = get_state_metadata  # type: ignore[attr-defined]
    cls.is_initial_state = is_initial_state  # type: ignore[attr-defined]
    cls.get_path_cost = get_path_cost  # type: ignore[attr-defined]
    cls.is_blocking = is_blocking  # type: ignore[attr-defined]
    cls.can_hide_state = can_hide_state  # type: ignore[attr-defined]


def is_state(obj: Any) -> bool:
    """Check if an object is a Qontinui state.

    Args:
        obj: Object to check

    Returns:
        True if object is decorated with @state
    """
    # Check both class and instance
    target = obj if isinstance(obj, type) else type(obj)
    return hasattr(target, "_qontinui_state") and target._qontinui_state


def get_state_metadata(obj: Any) -> StateMetadata | None:
    """Get state metadata from a decorated class or instance.

    Args:
        obj: The state class or instance

    Returns:
        StateMetadata or None if not a state
    """
    if not is_state(obj):
        return None

    target = obj if isinstance(obj, type) else type(obj)
    return getattr(target, "_qontinui_state_metadata", None)


def get_state_name(obj: Any) -> str | None:
    """Get the name of a state.

    Args:
        obj: The state class or instance

    Returns:
        State name or None
    """
    metadata = get_state_metadata(obj)
    return metadata.name if metadata else None


def get_initial_states_for_profile(states: list[Any], profile: str = "default") -> list[Any]:
    """Get initial states for a specific profile.

    Args:
        states: List of state classes or instances
        profile: Profile name

    Returns:
        List of initial states for the profile
    """
    initial_states = []

    for state in states:
        metadata = get_state_metadata(state)
        if metadata and metadata.initial:
            # Check if state is initial for this profile
            if not metadata.profiles or profile in metadata.profiles:
                initial_states.append(state)

    # Sort by priority (higher priority first)
    def get_priority(s):
        metadata = get_state_metadata(s)
        return metadata.priority if metadata is not None else 0

    initial_states.sort(key=get_priority, reverse=True)

    return initial_states


def get_states_in_group(states: list[Any], group_name: str) -> list[Any]:
    """Get all states in a specific group.

    Args:
        states: List of state classes or instances
        group_name: Name of the group

    Returns:
        List of states in the group
    """
    group_states = []

    for state in states:
        metadata = get_state_metadata(state)
        if metadata and metadata.group == group_name:
            group_states.append(state)

    return group_states
