"""Transition set decorator for defining state transitions.

This module provides decorators for defining transitions between states with:
- Multi-state activation support
- Incoming and outgoing transition types
- Path cost and visibility control
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from qontinui.model.transition.enhanced_state_transition import StaysVisible

logger = logging.getLogger(__name__)


class TransitionType(Enum):
    """Type of transition."""

    OUTGOING = "outgoing"  # Transition from a state
    INCOMING = "incoming"  # Transition to a state
    BIDIRECTIONAL = "bidirectional"  # Both directions


@dataclass
class TransitionSetMetadata:
    """Metadata for a transition set."""

    name: str
    from_states: list[type]
    to_states: list[type]
    activate_all: bool = True
    exit_all: bool = False
    transition_type: TransitionType = TransitionType.OUTGOING
    path_cost: int = 1
    stays_visible: StaysVisible = StaysVisible.NONE
    description: str = ""


def transition_set(
    from_states: list[type] | type | None = None,
    to_states: list[type] | type | None = None,
    activate_all: bool = True,
    exit_all: bool = False,
    name: str = "",
    transition_type: TransitionType = TransitionType.OUTGOING,
    path_cost: int = 1,
    stays_visible: StaysVisible = StaysVisible.NONE,
    description: str = "",
) -> Callable[[type], type]:
    """Define transition sets with multi-state support.

    This decorator marks a class as a transition set that defines
    how to move between states. It supports Brobot's multi-state
    activation pattern where transitions can activate multiple
    states simultaneously.

    Args:
        from_states: State(s) this transition originates from.
                    Can be a single state class or a list of states.
        to_states: State(s) this transition leads to.
                  Can be a single state class or a list of states.
        activate_all: If True, ALL to_states are activated together.
                     If False, states are activated based on conditions.
        exit_all: If True, ALL from_states are exited.
                 If False, only the originating state is exited.
        name: Optional name for the transition set.
        transition_type: Type of transition (outgoing, incoming, bidirectional).
        path_cost: Cost for pathfinding (higher = less preferred).
        stays_visible: Visibility behavior after transition.
        description: Description of the transition's purpose.

    Returns:
        The decorated class with transition metadata attached.

    Example:
        @transition_set(
            from_states=MainMenuState,
            to_states=[ToolbarState, SidebarState, ContentState],
            activate_all=True,
            stays_visible=StaysVisible.FALSE
        )
        class OpenWorkspaceTransition:
            '''Opens the full workspace with all panels.'''

            def execute(self) -> bool:
                # Custom transition logic
                return True

        @transition_set(
            from_states=[EditorState, PreviewState],
            to_states=MainMenuState,
            exit_all=True
        )
        class ReturnToMenuTransition:
            '''Returns to main menu from any editor state.'''
            pass

        @transition_set(
            from_states=LoginState,
            to_states=DashboardState,
            transition_type=TransitionType.INCOMING,
            path_cost=0
        )
        class PostLoginTransition:
            '''Incoming transition that runs after reaching Dashboard.'''

            def on_enter(self):
                # Initialize dashboard data
                pass
    """

    def decorator(cls: type) -> type:
        # Normalize inputs to lists
        from_list = _normalize_to_list(from_states)
        to_list = _normalize_to_list(to_states)

        # Derive name if not provided
        transition_name = name or cls.__name__

        # Create metadata
        metadata = TransitionSetMetadata(
            name=transition_name,
            from_states=from_list,
            to_states=to_list,
            activate_all=activate_all,
            exit_all=exit_all,
            transition_type=transition_type,
            path_cost=path_cost,
            stays_visible=stays_visible,
            description=description or cls.__doc__ or "",
        )

        # Store metadata on the class
        cls._qontinui_transition_set = True  # type: ignore[attr-defined]
        cls._qontinui_transition_metadata = metadata  # type: ignore[attr-defined]

        # Log transition registration
        logger.debug(
            f"TransitionSet decorated: {transition_name} "
            f"({len(from_list)} -> {len(to_list)} states, "
            f"activate_all={activate_all})"
        )

        # Add helper methods
        _add_transition_methods(cls, metadata)

        return cls

    return decorator


def outgoing_transition(
    from_state: type, to_states: list[type] | type, **kwargs
) -> Callable[[type], type]:
    """Shorthand decorator for outgoing transitions.

    Args:
        from_state: Source state
        to_states: Target state(s)
        **kwargs: Additional transition_set parameters

    Example:
        @outgoing_transition(MainMenuState, [ToolbarState, SidebarState])
        class OpenToolsTransition:
            pass
    """
    return transition_set(
        from_states=from_state,
        to_states=to_states,
        transition_type=TransitionType.OUTGOING,
        **kwargs,
    )


def incoming_transition(
    to_state: type, from_states: list[type] | type | None = None, **kwargs
) -> Callable[[type], type]:
    """Shorthand decorator for incoming transitions.

    Incoming transitions execute when entering a state,
    useful for initialization logic.

    Args:
        to_state: Target state
        from_states: Optional source state(s)
        **kwargs: Additional transition_set parameters

    Example:
        @incoming_transition(DashboardState)
        class InitializeDashboardTransition:
            def on_enter(self):
                # Load dashboard data
                pass
    """
    return transition_set(
        from_states=from_states,
        to_states=to_state,
        transition_type=TransitionType.INCOMING,
        **kwargs,
    )


def _normalize_to_list(states: list[type] | type | None) -> list[type]:
    """Normalize state input to a list.

    Args:
        states: Single state or list of states

    Returns:
        List of states
    """
    if states is None:
        return []
    if isinstance(states, list):
        return states
    return [states]


def _add_transition_methods(cls: type, metadata: TransitionSetMetadata) -> None:
    """Add helper methods to the transition class.

    Args:
        cls: The transition class
        metadata: Transition metadata
    """

    def get_transition_name(self) -> str:
        """Get the transition name."""
        return metadata.name

    def get_transition_metadata(self) -> TransitionSetMetadata:
        """Get the transition metadata."""
        return metadata

    def activates_states(self) -> list[type]:
        """Get states that will be activated."""
        if metadata.activate_all:
            return metadata.to_states
        return []

    def exits_states(self) -> list[type]:
        """Get states that will be exited."""
        if metadata.exit_all:
            return metadata.from_states
        return []

    def get_path_cost(self) -> int:
        """Get the path cost for this transition."""
        return metadata.path_cost

    def is_incoming(self) -> bool:
        """Check if this is an incoming transition."""
        return metadata.transition_type in [TransitionType.INCOMING, TransitionType.BIDIRECTIONAL]

    def is_outgoing(self) -> bool:
        """Check if this is an outgoing transition."""
        return metadata.transition_type in [TransitionType.OUTGOING, TransitionType.BIDIRECTIONAL]

    # Add methods to the class
    cls.get_transition_name = get_transition_name  # type: ignore[attr-defined]
    cls.get_transition_metadata = get_transition_metadata  # type: ignore[attr-defined]
    cls.activates_states = activates_states  # type: ignore[attr-defined]
    cls.exits_states = exits_states  # type: ignore[attr-defined]
    cls.get_path_cost = get_path_cost  # type: ignore[attr-defined]
    cls.is_incoming = is_incoming  # type: ignore[attr-defined]
    cls.is_outgoing = is_outgoing  # type: ignore[attr-defined]


def is_transition_set(obj: Any) -> bool:
    """Check if an object is a transition set.

    Args:
        obj: Object to check

    Returns:
        True if object is decorated with @transition_set
    """
    target = obj if isinstance(obj, type) else type(obj)
    return hasattr(target, "_qontinui_transition_set") and target._qontinui_transition_set


def get_transition_metadata(obj: Any) -> TransitionSetMetadata | None:
    """Get transition metadata from a decorated class or instance.

    Args:
        obj: The transition class or instance

    Returns:
        TransitionSetMetadata or None
    """
    if not is_transition_set(obj):
        return None

    target = obj if isinstance(obj, type) else type(obj)
    return getattr(target, "_qontinui_transition_metadata", None)


def get_transitions_from_state(transitions: list[Any], state_class: type) -> list[Any]:
    """Get all transitions originating from a state.

    Args:
        transitions: List of transition classes or instances
        state_class: The state class to find transitions from

    Returns:
        List of transitions from the state
    """
    matching_transitions = []

    for transition in transitions:
        metadata = get_transition_metadata(transition)
        if metadata and state_class in metadata.from_states:
            matching_transitions.append(transition)

    return matching_transitions


def get_transitions_to_state(transitions: list[Any], state_class: type) -> list[Any]:
    """Get all transitions leading to a state.

    Args:
        transitions: List of transition classes or instances
        state_class: The state class to find transitions to

    Returns:
        List of transitions to the state
    """
    matching_transitions = []

    for transition in transitions:
        metadata = get_transition_metadata(transition)
        if metadata and state_class in metadata.to_states:
            matching_transitions.append(transition)

    return matching_transitions


def get_incoming_transitions(transitions: list[Any], state_class: type) -> list[Any]:
    """Get incoming transitions for a state.

    Args:
        transitions: List of transition classes or instances
        state_class: The state class

    Returns:
        List of incoming transitions
    """
    incoming = []

    for transition in transitions:
        metadata = get_transition_metadata(transition)
        if (
            metadata
            and metadata.transition_type in [TransitionType.INCOMING, TransitionType.BIDIRECTIONAL]
            and state_class in metadata.to_states
        ):
            incoming.append(transition)

    return incoming
