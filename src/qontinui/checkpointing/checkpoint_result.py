"""Checkpoint data models for capturing automation state."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class CheckpointTrigger(Enum):
    """Enum for checkpoint trigger reasons."""

    MANUAL = "manual"  # Explicit checkpoint action
    TRANSITION_COMPLETE = "transition"  # After state transition
    TRANSITION_FAILURE = "transition_failure"  # Failed state transition
    TERMINAL_FAILURE = "terminal_failure"  # Non-recoverable action failure


@dataclass(frozen=True)
class TextRegionData:
    """OCR text region with position information.

    Immutable snapshot of detected text with bounding box coordinates.
    """

    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Get region bounds as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    @property
    def center(self) -> tuple[int, int]:
        """Get center point of text region."""
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass(frozen=True)
class CheckpointData:
    """Captured checkpoint data with screenshot and OCR information.

    Immutable snapshot of automation state at a specific point in time.
    """

    name: str
    timestamp: datetime
    screenshot_path: str | None
    ocr_text: str  # Full extracted text
    text_regions: tuple[TextRegionData, ...]  # Individual text regions
    trigger: CheckpointTrigger
    action_context: str | None  # Which action triggered it
    metadata: dict[str, Any] | None  # Additional context

    @property
    def has_screenshot(self) -> bool:
        """Check if checkpoint has a screenshot."""
        return self.screenshot_path is not None

    @property
    def region_count(self) -> int:
        """Get number of detected text regions."""
        return len(self.text_regions)

    def get_regions_containing(
        self, text: str, case_sensitive: bool = False
    ) -> list[TextRegionData]:
        """Find text regions containing specific text.

        Args:
            text: Text to search for
            case_sensitive: Whether search is case-sensitive

        Returns:
            List of matching text regions
        """
        search_text = text if case_sensitive else text.lower()
        matching_regions = []

        for region in self.text_regions:
            region_text = region.text if case_sensitive else region.text.lower()
            if search_text in region_text:
                matching_regions.append(region)

        return matching_regions
