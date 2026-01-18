"""Visual validation for action success verification.

This module provides screenshot-based validation to confirm that actions
succeeded by detecting visual changes on screen.

Key Features:
- Compare before/after screenshots for changes
- Check for specific element appearance/disappearance
- Monitor specific regions for changes
- Find and report changed regions for debugging

Example:
    >>> from qontinui.validation import VisualValidator, ExpectedChange, ChangeType
    >>>
    >>> validator = VisualValidator()
    >>>
    >>> # Validate that something changed
    >>> result = validator.validate(pre_screenshot, post_screenshot)
    >>> if not result.success:
    >>>     print(f"Action may have failed: {result.message}")
    >>>
    >>> # Validate specific element appeared
    >>> expected = ExpectedChange(
    >>>     type=ChangeType.ELEMENT_APPEARS,
    >>>     pattern=dialog_pattern,
    >>>     similarity_threshold=0.8,
    >>> )
    >>> result = validator.validate(pre_screenshot, post_screenshot, expected)
"""

from .validation_types import (
    ChangedRegion,
    ChangeType,
    ExpectedChange,
    ValidationResult,
    VisualDiff,
)
from .visual_validator import (
    VisualValidator,
    get_visual_validator,
    set_visual_validator,
)

__all__ = [
    # Main class
    "VisualValidator",
    "get_visual_validator",
    "set_visual_validator",
    # Types
    "ValidationResult",
    "VisualDiff",
    "ExpectedChange",
    "ChangeType",
    "ChangedRegion",
]
