"""Spatial relationship assertions.

Provides assertions for element positioning, alignment,
and spatial relationships between elements.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.common import utc_now
from qontinui_schemas.testing.assertions import (
    AssertionResult,
    AssertionStatus,
    AssertionType,
    BoundingBox,
)

if TYPE_CHECKING:
    from qontinui.vision.verification.config import VisionConfig
    from qontinui.vision.verification.locators.base import BaseLocator

logger = logging.getLogger(__name__)


class Alignment(str, Enum):
    """Alignment types."""

    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    TOP = "top"
    BOTTOM = "bottom"
    MIDDLE = "middle"


class Direction(str, Enum):
    """Spatial direction."""

    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"


@dataclass
class SpatialRelation:
    """Spatial relationship between two elements."""

    direction: Direction | None
    distance: float
    overlap: bool
    overlap_area: int
    aligned_horizontally: bool
    aligned_vertically: bool


class SpatialAssertion:
    """Assertions for spatial relationships between elements.

    Usage:
        assertion = SpatialAssertion(element_locator, config)

        # Direction assertions
        result = await assertion.to_be_above(reference_locator, screenshot)
        result = await assertion.to_be_below(reference_locator, screenshot)
        result = await assertion.to_be_left_of(reference_locator, screenshot)
        result = await assertion.to_be_right_of(reference_locator, screenshot)

        # Alignment assertions
        result = await assertion.to_be_aligned_with(reference_locator, "left", screenshot)
        result = await assertion.to_be_horizontally_aligned(reference_locator, screenshot)
        result = await assertion.to_be_vertically_aligned(reference_locator, screenshot)

        # Distance assertions
        result = await assertion.to_be_near(reference_locator, 50, screenshot)
        result = await assertion.to_be_far_from(reference_locator, 100, screenshot)

        # Containment assertions
        result = await assertion.to_contain(child_locator, screenshot)
        result = await assertion.to_be_contained_in(parent_locator, screenshot)

        # Overlap assertions
        result = await assertion.to_overlap_with(other_locator, screenshot)
        result = await assertion.to_not_overlap_with(other_locator, screenshot)
    """

    def __init__(
        self,
        locator: "BaseLocator",
        config: "VisionConfig | None" = None,
    ) -> None:
        """Initialize spatial assertion.

        Args:
            locator: Locator for the target element.
            config: Vision configuration.
        """
        self._locator = locator
        self._config = config

    def _get_center(self, bounds: BoundingBox) -> tuple[float, float]:
        """Get center point of bounding box.

        Args:
            bounds: Bounding box.

        Returns:
            Center (x, y) coordinates.
        """
        return (
            bounds.x + bounds.width / 2,
            bounds.y + bounds.height / 2,
        )

    def _calculate_distance(self, b1: BoundingBox, b2: BoundingBox) -> float:
        """Calculate minimum distance between bounding boxes.

        Args:
            b1: First bounding box.
            b2: Second bounding box.

        Returns:
            Minimum distance (0 if overlapping).
        """
        # Calculate gaps in each direction
        left_gap = b2.x - (b1.x + b1.width)  # b1 to left of b2
        right_gap = b1.x - (b2.x + b2.width)  # b1 to right of b2
        top_gap = b2.y - (b1.y + b1.height)  # b1 above b2
        bottom_gap = b1.y - (b2.y + b2.height)  # b1 below b2

        horizontal_gap = max(left_gap, right_gap, 0)
        vertical_gap = max(top_gap, bottom_gap, 0)

        # Euclidean distance
        return float(np.sqrt(horizontal_gap**2 + vertical_gap**2))

    def _calculate_overlap_area(self, b1: BoundingBox, b2: BoundingBox) -> int:
        """Calculate overlap area between bounding boxes.

        Args:
            b1: First bounding box.
            b2: Second bounding box.

        Returns:
            Overlap area in pixels.
        """
        x_overlap = max(
            0,
            min(b1.x + b1.width, b2.x + b2.width) - max(b1.x, b2.x),
        )
        y_overlap = max(
            0,
            min(b1.y + b1.height, b2.y + b2.height) - max(b1.y, b2.y),
        )
        return int(x_overlap * y_overlap)

    def _get_spatial_relation(
        self,
        target: BoundingBox,
        reference: BoundingBox,
    ) -> SpatialRelation:
        """Analyze spatial relationship between two elements.

        Args:
            target: Target element bounds.
            reference: Reference element bounds.

        Returns:
            Spatial relationship analysis.
        """
        # Centers
        t_center = self._get_center(target)
        r_center = self._get_center(reference)

        # Determine primary direction
        dx = t_center[0] - r_center[0]
        dy = t_center[1] - r_center[1]

        direction: Direction | None = None
        if abs(dy) > abs(dx):
            direction = Direction.ABOVE if dy < 0 else Direction.BELOW
        elif abs(dx) > abs(dy):
            direction = Direction.LEFT_OF if dx < 0 else Direction.RIGHT_OF

        # Distance
        distance = self._calculate_distance(target, reference)

        # Overlap
        overlap_area = self._calculate_overlap_area(target, reference)
        overlap = overlap_area > 0

        # Alignment (within 5px tolerance)
        tolerance = 5
        aligned_horizontally = abs(t_center[1] - r_center[1]) <= tolerance
        aligned_vertically = abs(t_center[0] - r_center[0]) <= tolerance

        return SpatialRelation(
            direction=direction,
            distance=distance,
            overlap=overlap,
            overlap_area=overlap_area,
            aligned_horizontally=aligned_horizontally,
            aligned_vertically=aligned_vertically,
        )

    async def to_be_above(
        self,
        reference: "BaseLocator",
        screenshot: NDArray[np.uint8],
        min_gap: int = 0,
    ) -> AssertionResult:
        """Assert element is above reference element.

        Args:
            reference: Reference element locator.
            screenshot: Screenshot to analyze.
            min_gap: Minimum vertical gap required.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        start_time = time.monotonic()

        target_match = await self._locator.find(screenshot)
        ref_match = await reference.find(screenshot)

        if target_match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            completed_at = utc_now()
            return AssertionResult(
                assertion_id="spatial_above",
                assertion_method="to_be_above",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Target element not found",
                expected_value="above reference",
                actual_value="target not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        if ref_match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            completed_at = utc_now()
            return AssertionResult(
                assertion_id="spatial_above",
                assertion_method="to_be_above",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Reference element not found",
                expected_value="above reference",
                actual_value="reference not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        t = target_match.bounds
        r = ref_match.bounds

        # Target bottom should be above reference top
        is_above = t.y + t.height <= r.y
        gap = r.y - (t.y + t.height) if is_above else 0
        has_min_gap = gap >= min_gap

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        completed_at = utc_now()

        if is_above and has_min_gap:
            return AssertionResult(
                assertion_id="spatial_above",
                assertion_method="to_be_above",
                status=AssertionStatus.PASSED,
                started_at=started_at,
                completed_at=completed_at,
                expected_value="above reference",
                actual_value=f"above by {gap}px",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            t_center = self._get_center(t)
            r_center = self._get_center(r)
            return AssertionResult(
                assertion_id="spatial_above",
                assertion_method="to_be_above",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message=f"Element not above reference (target y: {t_center[1]}, ref y: {r_center[1]})",
                expected_value="above reference",
                actual_value="not above",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_be_below(
        self,
        reference: "BaseLocator",
        screenshot: NDArray[np.uint8],
        min_gap: int = 0,
    ) -> AssertionResult:
        """Assert element is below reference element.

        Args:
            reference: Reference element locator.
            screenshot: Screenshot to analyze.
            min_gap: Minimum vertical gap required.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        start_time = time.monotonic()

        target_match = await self._locator.find(screenshot)
        ref_match = await reference.find(screenshot)

        if target_match is None or ref_match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            completed_at = utc_now()
            return AssertionResult(
                assertion_id="spatial_below",
                assertion_method="to_be_below",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Element(s) not found",
                expected_value="below reference",
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        t = target_match.bounds
        r = ref_match.bounds

        is_below = t.y >= r.y + r.height
        gap = t.y - (r.y + r.height) if is_below else 0
        has_min_gap = gap >= min_gap

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        completed_at = utc_now()

        if is_below and has_min_gap:
            return AssertionResult(
                assertion_id="spatial_below",
                assertion_method="to_be_below",
                status=AssertionStatus.PASSED,
                started_at=started_at,
                completed_at=completed_at,
                expected_value="below reference",
                actual_value=f"below by {gap}px",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="spatial_below",
                assertion_method="to_be_below",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Element not below reference",
                expected_value="below reference",
                actual_value="not below",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_be_left_of(
        self,
        reference: "BaseLocator",
        screenshot: NDArray[np.uint8],
        min_gap: int = 0,
    ) -> AssertionResult:
        """Assert element is left of reference element.

        Args:
            reference: Reference element locator.
            screenshot: Screenshot to analyze.
            min_gap: Minimum horizontal gap required.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        start_time = time.monotonic()

        target_match = await self._locator.find(screenshot)
        ref_match = await reference.find(screenshot)

        if target_match is None or ref_match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            completed_at = utc_now()
            return AssertionResult(
                assertion_id="spatial_left_of",
                assertion_method="to_be_left_of",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Element(s) not found",
                expected_value="left of reference",
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        t = target_match.bounds
        r = ref_match.bounds

        is_left = t.x + t.width <= r.x
        gap = r.x - (t.x + t.width) if is_left else 0
        has_min_gap = gap >= min_gap

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        completed_at = utc_now()

        if is_left and has_min_gap:
            return AssertionResult(
                assertion_id="spatial_left_of",
                assertion_method="to_be_left_of",
                status=AssertionStatus.PASSED,
                started_at=started_at,
                completed_at=completed_at,
                expected_value="left of reference",
                actual_value=f"left by {gap}px",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="spatial_left_of",
                assertion_method="to_be_left_of",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Element not left of reference",
                expected_value="left of reference",
                actual_value="not left",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_be_right_of(
        self,
        reference: "BaseLocator",
        screenshot: NDArray[np.uint8],
        min_gap: int = 0,
    ) -> AssertionResult:
        """Assert element is right of reference element.

        Args:
            reference: Reference element locator.
            screenshot: Screenshot to analyze.
            min_gap: Minimum horizontal gap required.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        start_time = time.monotonic()

        target_match = await self._locator.find(screenshot)
        ref_match = await reference.find(screenshot)

        if target_match is None or ref_match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            completed_at = utc_now()
            return AssertionResult(
                assertion_id="spatial_right_of",
                assertion_method="to_be_right_of",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Element(s) not found",
                expected_value="right of reference",
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        t = target_match.bounds
        r = ref_match.bounds

        is_right = t.x >= r.x + r.width
        gap = t.x - (r.x + r.width) if is_right else 0
        has_min_gap = gap >= min_gap

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        completed_at = utc_now()

        if is_right and has_min_gap:
            return AssertionResult(
                assertion_id="spatial_right_of",
                assertion_method="to_be_right_of",
                status=AssertionStatus.PASSED,
                started_at=started_at,
                completed_at=completed_at,
                expected_value="right of reference",
                actual_value=f"right by {gap}px",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="spatial_right_of",
                assertion_method="to_be_right_of",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Element not right of reference",
                expected_value="right of reference",
                actual_value="not right",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_be_aligned_with(
        self,
        reference: "BaseLocator",
        alignment: str | Alignment,
        screenshot: NDArray[np.uint8],
        tolerance: int = 5,
    ) -> AssertionResult:
        """Assert element is aligned with reference.

        Args:
            reference: Reference element locator.
            alignment: Alignment type (left, right, center, top, bottom, middle).
            screenshot: Screenshot to analyze.
            tolerance: Alignment tolerance in pixels.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        start_time = time.monotonic()

        if isinstance(alignment, str):
            alignment = Alignment(alignment)

        target_match = await self._locator.find(screenshot)
        ref_match = await reference.find(screenshot)

        if target_match is None or ref_match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            completed_at = utc_now()
            return AssertionResult(
                assertion_id="spatial_aligned",
                assertion_method="to_be_aligned_with",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Element(s) not found",
                expected_value=f"aligned {alignment.value}",
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        t = target_match.bounds
        r = ref_match.bounds

        # Calculate alignment positions
        t_center = self._get_center(t)
        r_center = self._get_center(r)

        aligned = False
        actual_diff: float = 0.0

        if alignment == Alignment.LEFT:
            actual_diff = float(abs(t.x - r.x))
            aligned = actual_diff <= tolerance
        elif alignment == Alignment.RIGHT:
            actual_diff = float(abs((t.x + t.width) - (r.x + r.width)))
            aligned = actual_diff <= tolerance
        elif alignment == Alignment.CENTER:
            actual_diff = float(abs(t_center[0] - r_center[0]))
            aligned = actual_diff <= tolerance
        elif alignment == Alignment.TOP:
            actual_diff = float(abs(t.y - r.y))
            aligned = actual_diff <= tolerance
        elif alignment == Alignment.BOTTOM:
            actual_diff = float(abs((t.y + t.height) - (r.y + r.height)))
            aligned = actual_diff <= tolerance
        elif alignment == Alignment.MIDDLE:
            actual_diff = float(abs(t_center[1] - r_center[1]))
            aligned = actual_diff <= tolerance

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        completed_at = utc_now()

        if aligned:
            return AssertionResult(
                assertion_id="spatial_aligned",
                assertion_method="to_be_aligned_with",
                status=AssertionStatus.PASSED,
                started_at=started_at,
                completed_at=completed_at,
                expected_value=f"aligned {alignment.value}",
                actual_value=f"diff {actual_diff}px",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="spatial_aligned",
                assertion_method="to_be_aligned_with",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message=f"Not aligned {alignment.value} (diff: {actual_diff}px)",
                expected_value=f"aligned {alignment.value}",
                actual_value=f"diff {actual_diff}px",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_be_horizontally_aligned(
        self,
        reference: "BaseLocator",
        screenshot: NDArray[np.uint8],
        tolerance: int = 5,
    ) -> AssertionResult:
        """Assert element is horizontally aligned (same Y center).

        Args:
            reference: Reference element locator.
            screenshot: Screenshot to analyze.
            tolerance: Alignment tolerance in pixels.

        Returns:
            Assertion result.
        """
        return await self.to_be_aligned_with(reference, Alignment.MIDDLE, screenshot, tolerance)

    async def to_be_vertically_aligned(
        self,
        reference: "BaseLocator",
        screenshot: NDArray[np.uint8],
        tolerance: int = 5,
    ) -> AssertionResult:
        """Assert element is vertically aligned (same X center).

        Args:
            reference: Reference element locator.
            screenshot: Screenshot to analyze.
            tolerance: Alignment tolerance in pixels.

        Returns:
            Assertion result.
        """
        return await self.to_be_aligned_with(reference, Alignment.CENTER, screenshot, tolerance)

    async def to_be_near(
        self,
        reference: "BaseLocator",
        max_distance: int,
        screenshot: NDArray[np.uint8],
    ) -> AssertionResult:
        """Assert element is within distance of reference.

        Args:
            reference: Reference element locator.
            max_distance: Maximum distance in pixels.
            screenshot: Screenshot to analyze.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        start_time = time.monotonic()

        target_match = await self._locator.find(screenshot)
        ref_match = await reference.find(screenshot)

        if target_match is None or ref_match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            completed_at = utc_now()
            return AssertionResult(
                assertion_id="spatial_near",
                assertion_method="to_be_near",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Element(s) not found",
                expected_value=f"within {max_distance}px",
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        distance = self._calculate_distance(target_match.bounds, ref_match.bounds)
        is_near = distance <= max_distance

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        completed_at = utc_now()

        if is_near:
            return AssertionResult(
                assertion_id="spatial_near",
                assertion_method="to_be_near",
                status=AssertionStatus.PASSED,
                started_at=started_at,
                completed_at=completed_at,
                expected_value=f"within {max_distance}px",
                actual_value=f"{distance:.0f}px",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="spatial_near",
                assertion_method="to_be_near",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message=f"Element too far from reference ({distance:.0f}px > {max_distance}px)",
                expected_value=f"within {max_distance}px",
                actual_value=f"{distance:.0f}px",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_overlap_with(
        self,
        other: "BaseLocator",
        screenshot: NDArray[np.uint8],
        min_overlap_area: int = 1,
    ) -> AssertionResult:
        """Assert element overlaps with another element.

        Args:
            other: Other element locator.
            screenshot: Screenshot to analyze.
            min_overlap_area: Minimum overlap area in pixels.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        start_time = time.monotonic()

        target_match = await self._locator.find(screenshot)
        other_match = await other.find(screenshot)

        if target_match is None or other_match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            completed_at = utc_now()
            return AssertionResult(
                assertion_id="spatial_overlap",
                assertion_method="to_overlap_with",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Element(s) not found",
                expected_value="overlapping",
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        overlap_area = self._calculate_overlap_area(target_match.bounds, other_match.bounds)
        overlaps = overlap_area >= min_overlap_area

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        completed_at = utc_now()

        if overlaps:
            return AssertionResult(
                assertion_id="spatial_overlap",
                assertion_method="to_overlap_with",
                status=AssertionStatus.PASSED,
                started_at=started_at,
                completed_at=completed_at,
                expected_value="overlapping",
                actual_value=f"{overlap_area}px²",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="spatial_overlap",
                assertion_method="to_overlap_with",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Elements do not overlap",
                expected_value="overlapping",
                actual_value="no overlap",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_not_overlap_with(
        self,
        other: "BaseLocator",
        screenshot: NDArray[np.uint8],
    ) -> AssertionResult:
        """Assert element does not overlap with another element.

        Args:
            other: Other element locator.
            screenshot: Screenshot to analyze.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        start_time = time.monotonic()

        target_match = await self._locator.find(screenshot)
        other_match = await other.find(screenshot)

        if target_match is None or other_match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            completed_at = utc_now()
            return AssertionResult(
                assertion_id="spatial_no_overlap",
                assertion_method="to_not_overlap_with",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Element(s) not found",
                expected_value="not overlapping",
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        overlap_area = self._calculate_overlap_area(target_match.bounds, other_match.bounds)
        no_overlap = overlap_area == 0

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        completed_at = utc_now()

        if no_overlap:
            return AssertionResult(
                assertion_id="spatial_no_overlap",
                assertion_method="to_not_overlap_with",
                status=AssertionStatus.PASSED,
                started_at=started_at,
                completed_at=completed_at,
                expected_value="not overlapping",
                actual_value="no overlap",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="spatial_no_overlap",
                assertion_method="to_not_overlap_with",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message=f"Elements overlap ({overlap_area}px²)",
                expected_value="not overlapping",
                actual_value=f"{overlap_area}px²",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_contain(
        self,
        child: "BaseLocator",
        screenshot: NDArray[np.uint8],
    ) -> AssertionResult:
        """Assert element contains another element.

        Args:
            child: Child element locator.
            screenshot: Screenshot to analyze.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        start_time = time.monotonic()

        target_match = await self._locator.find(screenshot)
        child_match = await child.find(screenshot)

        if target_match is None or child_match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            completed_at = utc_now()
            return AssertionResult(
                assertion_id="spatial_contains",
                assertion_method="to_contain",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Element(s) not found",
                expected_value="contains child",
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        p = target_match.bounds  # Parent
        c = child_match.bounds  # Child

        contains = (
            c.x >= p.x
            and c.y >= p.y
            and c.x + c.width <= p.x + p.width
            and c.y + c.height <= p.y + p.height
        )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        completed_at = utc_now()

        if contains:
            return AssertionResult(
                assertion_id="spatial_contains",
                assertion_method="to_contain",
                status=AssertionStatus.PASSED,
                started_at=started_at,
                completed_at=completed_at,
                expected_value="contains child",
                actual_value="contained",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="spatial_contains",
                assertion_method="to_contain",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Element does not fully contain child",
                expected_value="contains child",
                actual_value="not contained",
                matches_found=0,
                duration_ms=elapsed_ms,
            )


__all__ = [
    "Alignment",
    "Direction",
    "SpatialAssertion",
    "SpatialRelation",
]
