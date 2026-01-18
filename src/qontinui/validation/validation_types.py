"""Type definitions for visual validation.

Defines data structures for validation results and expected changes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChangeType(Enum):
    """Type of expected visual change after an action."""

    ANY_CHANGE = "any_change"
    """Verify that something changed (default)."""

    ELEMENT_APPEARS = "element_appears"
    """Verify that a specific element becomes visible."""

    ELEMENT_DISAPPEARS = "element_disappears"
    """Verify that a specific element is no longer visible."""

    REGION_CHANGES = "region_changes"
    """Verify that a specific screen region changed."""

    NO_CHANGE = "no_change"
    """Verify that nothing changed (for read-only operations)."""


@dataclass
class ChangedRegion:
    """A region of the screen that changed."""

    x: int
    """X coordinate of region."""

    y: int
    """Y coordinate of region."""

    width: int
    """Width of region."""

    height: int
    """Height of region."""

    change_intensity: float
    """How much the region changed (0.0-1.0)."""


@dataclass
class VisualDiff:
    """Result of comparing two screenshots."""

    change_percentage: float
    """Percentage of pixels that changed (0.0-100.0)."""

    changed_pixel_count: int
    """Number of pixels that changed."""

    total_pixel_count: int
    """Total pixels in image."""

    changed_regions: list[ChangedRegion] = field(default_factory=list)
    """Contiguous regions that changed."""

    mean_change_intensity: float = 0.0
    """Average intensity of change across changed pixels."""


@dataclass
class ExpectedChange:
    """Definition of expected change after an action."""

    type: ChangeType
    """Type of change expected."""

    description: str = ""
    """Human-readable description of the expected change."""

    # For ELEMENT_APPEARS / ELEMENT_DISAPPEARS
    pattern: Any | None = None
    """Pattern to check for (numpy array or Pattern object)."""

    similarity_threshold: float = 0.8
    """Minimum similarity for pattern matching."""

    # For REGION_CHANGES
    region: tuple[int, int, int, int] | None = None
    """Region to check (x, y, width, height)."""

    min_change_threshold: float = 1.0
    """Minimum change percentage for REGION_CHANGES."""

    # For ANY_CHANGE
    min_any_change: float = 0.1
    """Minimum overall change for ANY_CHANGE type."""


@dataclass
class ValidationResult:
    """Result of validating an action."""

    success: bool
    """Whether validation passed."""

    message: str = ""
    """Human-readable result message."""

    diff: VisualDiff | None = None
    """Visual diff if computed."""

    expected: ExpectedChange | None = None
    """What was expected (if specified)."""

    actual_change_percentage: float = 0.0
    """Actual percentage of screen that changed."""

    validation_time_ms: float = 0.0
    """Time spent validating in milliseconds."""
