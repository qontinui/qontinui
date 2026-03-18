"""Data classes for multi-state GUI automation capture.

These types define the input/output contract for the multi-state capture
orchestrator that lives in the Python bridge (qontinui_executor.py).
The orchestrator walks through UI interactions, captures screenshots at
each step, diffs element sets to find new elements per state, and builds
a complete QontinuiConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qontinui.discovery.element_image_pipeline import ExtractedElementImage


@dataclass
class CaptureInteraction:
    """A single interaction step in a multi-state capture sequence.

    Attributes:
        action_type: The type of action to perform. "initial" means no action,
            just capture the current state. "click" and "scroll" interact with
            the target element before capturing.
        target: Element ID to act on. Required for "click" and "scroll",
            ignored for "initial".
        state_name: Human-readable name for the state revealed by this interaction.
        wait_seconds: Seconds to wait after performing the action before
            capturing the snapshot. Allows UI animations to settle.
    """

    action_type: str
    state_name: str
    target: str | None = None
    wait_seconds: float = 1.0


@dataclass
class CapturedState:
    """Result of capturing a single state during multi-state exploration.

    Each state contains only the elements that are NEW compared to all
    previously seen elements. This ensures no element image is duplicated
    across states.

    Attributes:
        state_id: Unique identifier for this state (e.g., "state-0").
        state_name: Human-readable name from the interaction definition.
        new_element_ids: Element IDs that first appeared in this state.
        element_images: Mapping of element ID to its extracted image data.
        interaction: The interaction that triggered this state, or None
            for the initial state.
    """

    state_id: str
    state_name: str
    new_element_ids: list[str]
    element_images: dict[str, ExtractedElementImage] = field(default_factory=dict)
    interaction: CaptureInteraction | None = None


@dataclass
class MultiStateCaptureResult:
    """Complete result of a multi-state capture session.

    Attributes:
        states: List of captured states, each with only its new elements.
        total_elements_seen: Total unique element IDs encountered.
        interactions_processed: Number of interactions that were executed.
    """

    states: list[CapturedState]
    total_elements_seen: int
    interactions_processed: int
