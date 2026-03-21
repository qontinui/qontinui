"""Bridge between qontinui's state machine and Aria-UI's action history format.

Converts recent state machine transitions into the (screenshot, action_description)
pairs that AriaUIContextClient expects for context-aware grounding.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..model.state.state_memory import StateMemory

logger = logging.getLogger(__name__)


class ScreenshotProvider(Protocol):
    """Protocol for retrieving screenshots by state name.

    Implementations should load the screenshot bytes (PNG) for a given
    state, typically from cached state fingerprints or a screenshot store.
    """

    def get_screenshot(self, state_name: str) -> bytes | None:
        """Get PNG screenshot bytes for a state.

        Args:
            state_name: Name of the state.

        Returns:
            PNG image bytes, or None if not available.
        """
        ...


def build_aria_ui_context(
    state_memory: StateMemory,
    screenshot_provider: ScreenshotProvider,
    max_history: int = 3,
) -> list[tuple[bytes, str]]:
    """Convert recent state machine transitions to Aria-UI context format.

    Reads the last N transitions from StateMemory, retrieves the corresponding
    screenshot for each source state, and formats them as (screenshot, action)
    pairs suitable for AriaUIContextClient.find_element_with_history().

    Args:
        state_memory: StateMemory instance with transition history.
        screenshot_provider: Provides screenshot bytes by state name.
        max_history: Maximum number of history entries to return.

    Returns:
        List of (screenshot_bytes, action_description) tuples in
        chronological order. May be shorter than max_history if screenshots
        are unavailable.
    """
    transitions = state_memory.get_transition_history(limit=max_history)
    context: list[tuple[bytes, str]] = []

    for transition in transitions:
        from_name = transition.from_state.name if transition.from_state else None
        to_name = transition.to_state.name if transition.to_state else None

        if not from_name:
            continue

        screenshot = screenshot_provider.get_screenshot(from_name)
        if screenshot is None:
            logger.debug(f"No screenshot available for state '{from_name}', skipping")
            continue

        description = _describe_transition(from_name, to_name, transition)
        context.append((screenshot, description))

    return context


def build_aria_ui_context_from_records(
    transition_records: list[dict[str, Any]],
    screenshot_provider: ScreenshotProvider,
    max_history: int = 3,
) -> list[tuple[bytes, str]]:
    """Build Aria-UI context from raw transition record dicts.

    Alternative to build_aria_ui_context() for cases where you have
    transition records from the StateTransitionAspect or other sources
    rather than a StateMemory instance.

    Args:
        transition_records: List of dicts with 'from_state', 'to_state',
            and optionally 'action' or 'transition_type' keys.
        screenshot_provider: Provides screenshot bytes by state name.
        max_history: Maximum number of history entries.

    Returns:
        List of (screenshot_bytes, action_description) tuples.
    """
    recent = transition_records[-max_history:]
    context: list[tuple[bytes, str]] = []

    for record in recent:
        from_state = record.get("from_state")
        to_state = record.get("to_state")

        if not from_state:
            continue

        screenshot = screenshot_provider.get_screenshot(from_state)
        if screenshot is None:
            logger.debug(f"No screenshot for state '{from_state}', skipping")
            continue

        action = record.get("action") or record.get("transition_type") or "transition"
        description = f"Transitioned from '{from_state}' to '{to_state}' via '{action}'"
        context.append((screenshot, description))

    return context


def _describe_transition(
    from_name: str,
    to_name: str | None,
    transition: Any,
) -> str:
    """Build a natural language description of a transition.

    Args:
        from_name: Source state name.
        to_name: Target state name (may be None).
        transition: StateTransition object.

    Returns:
        Human-readable description string.
    """
    # Try to get the transition type for a richer description
    t_type = getattr(transition, "transition_type", None)
    type_str = t_type.value if t_type else "action"

    to_part = f" to '{to_name}'" if to_name else ""
    return f"Transitioned from '{from_name}'{to_part} via {type_str}"
