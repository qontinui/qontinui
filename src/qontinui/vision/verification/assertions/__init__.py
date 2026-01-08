"""Vision assertion implementations.

Provides dedicated assertion classes for different verification types:
- VisibilityAssertion - to_be_visible, to_be_hidden
- TextAssertion - to_have_text, to_contain_text, to_match_text
- CountAssertion - to_have_count, to_have_count_at_least, to_be_empty
- ScreenshotAssertion - to_match_screenshot (visual regression)
- AnimationAssertion - to_stop_animating, to_be_stable
- AttributeAssertion - to_have_color, to_have_size, to_have_position
- SpatialAssertion - to_be_above, to_be_aligned_with, to_overlap_with
- StateAssertion - to_be_enabled, to_be_disabled, to_be_checked

These classes integrate with the detection engines and environment
for accurate verification.
"""

from qontinui.vision.verification.assertions.animation import (
    AnimationAssertion,
    AnimationDetector,
    AnimationResult,
    StabilityDetector,
    StabilityResult,
)
from qontinui.vision.verification.assertions.attributes import (
    AttributeAssertion,
    Color,
    Position,
    Size,
)
from qontinui.vision.verification.assertions.count import CountAssertion
from qontinui.vision.verification.assertions.screenshot import (
    ComparisonResult,
    ScreenshotAssertion,
    ScreenshotComparator,
    get_screenshot_comparator,
)
from qontinui.vision.verification.assertions.spatial import (
    Alignment,
    Direction,
    SpatialAssertion,
    SpatialRelation,
)
from qontinui.vision.verification.assertions.state import (
    ElementState,
    StateAssertion,
    StateDetectionResult,
    StateDetector,
)
from qontinui.vision.verification.assertions.text import TextAssertion
from qontinui.vision.verification.assertions.visibility import VisibilityAssertion

__all__ = [
    # Visibility
    "VisibilityAssertion",
    # Text
    "TextAssertion",
    # Count
    "CountAssertion",
    # Screenshot
    "ComparisonResult",
    "ScreenshotAssertion",
    "ScreenshotComparator",
    "get_screenshot_comparator",
    # Animation
    "AnimationAssertion",
    "AnimationDetector",
    "AnimationResult",
    "StabilityDetector",
    "StabilityResult",
    # Attributes
    "AttributeAssertion",
    "Color",
    "Position",
    "Size",
    # Spatial
    "Alignment",
    "Direction",
    "SpatialAssertion",
    "SpatialRelation",
    # State
    "ElementState",
    "StateAssertion",
    "StateDetectionResult",
    "StateDetector",
]
