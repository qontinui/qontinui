"""Attribute assertions for element visual properties.

Provides assertions for color, size, position, and other
visual attributes of elements.
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.assertions import (
    AssertionResult,
    AssertionStatus,
    AssertionType,
    BoundingBox,
)

if TYPE_CHECKING:
    from qontinui_schemas.testing.environment import GUIEnvironment

    from qontinui.vision.verification.config import VisionConfig
    from qontinui.vision.verification.locators.base import BaseLocator

logger = logging.getLogger(__name__)


@dataclass
class Color:
    """RGB color representation."""

    r: int
    g: int
    b: int

    @classmethod
    def from_hex(cls, hex_color: str) -> "Color":
        """Create color from hex string.

        Args:
            hex_color: Hex color string (e.g., '#FF0000' or 'FF0000').

        Returns:
            Color instance.
        """
        hex_color = hex_color.lstrip("#")
        return cls(
            r=int(hex_color[0:2], 16),
            g=int(hex_color[2:4], 16),
            b=int(hex_color[4:6], 16),
        )

    @classmethod
    def from_bgr(cls, bgr: tuple[int, int, int]) -> "Color":
        """Create color from BGR tuple.

        Args:
            bgr: BGR tuple from OpenCV.

        Returns:
            Color instance.
        """
        return cls(r=bgr[2], g=bgr[1], b=bgr[0])

    def to_hex(self) -> str:
        """Convert to hex string.

        Returns:
            Hex color string.
        """
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def to_bgr(self) -> tuple[int, int, int]:
        """Convert to BGR tuple for OpenCV.

        Returns:
            BGR tuple.
        """
        return (self.b, self.g, self.r)

    def distance(self, other: "Color") -> float:
        """Calculate Euclidean distance to another color.

        Args:
            other: Other color.

        Returns:
            Distance (0-441.67 for RGB space).
        """
        return np.sqrt((self.r - other.r) ** 2 + (self.g - other.g) ** 2 + (self.b - other.b) ** 2)


@dataclass
class Size:
    """Size representation."""

    width: int
    height: int

    @property
    def area(self) -> int:
        """Calculate area."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width/height)."""
        return self.width / self.height if self.height > 0 else 0


@dataclass
class Position:
    """Position representation."""

    x: int
    y: int

    def distance_to(self, other: "Position") -> float:
        """Calculate distance to another position.

        Args:
            other: Other position.

        Returns:
            Euclidean distance.
        """
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class AttributeAssertion:
    """Assertions for element visual attributes.

    Usage:
        assertion = AttributeAssertion(locator, config)

        # Color assertions
        result = await assertion.to_have_color("#FF0000", screenshot)
        result = await assertion.to_have_background_color("#FFFFFF", screenshot)

        # Size assertions
        result = await assertion.to_have_size(100, 50, screenshot)
        result = await assertion.to_have_width(100, screenshot)
        result = await assertion.to_have_height(50, screenshot)

        # Position assertions
        result = await assertion.to_have_position(200, 100, screenshot)
        result = await assertion.to_be_within_bounds(region, screenshot)
    """

    def __init__(
        self,
        locator: "BaseLocator",
        config: "VisionConfig | None" = None,
        environment: "GUIEnvironment | None" = None,
    ) -> None:
        """Initialize attribute assertion.

        Args:
            locator: Locator for finding the element.
            config: Vision configuration.
            environment: GUI environment for color reference.
        """
        self._locator = locator
        self._config = config
        self._environment = environment

    def _extract_region(
        self,
        screenshot: NDArray[np.uint8],
        bounds: BoundingBox,
    ) -> NDArray[np.uint8]:
        """Extract region from screenshot.

        Args:
            screenshot: Full screenshot.
            bounds: Region bounds.

        Returns:
            Cropped region.
        """
        return screenshot[
            bounds.y : bounds.y + bounds.height,
            bounds.x : bounds.x + bounds.width,
        ]

    def _get_dominant_color(self, region: NDArray[np.uint8]) -> Color:
        """Get dominant color in region.

        Args:
            region: Image region.

        Returns:
            Dominant color.
        """
        # Reshape to list of pixels
        pixels = region.reshape(-1, 3)

        # Use k-means to find dominant color
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        dominant = centers[0].astype(int)
        return Color.from_bgr(tuple(dominant))

    def _get_average_color(self, region: NDArray[np.uint8]) -> Color:
        """Get average color in region.

        Args:
            region: Image region.

        Returns:
            Average color.
        """
        avg = cv2.mean(region)[:3]
        return Color.from_bgr(tuple(int(c) for c in avg))

    def _get_border_color(
        self,
        region: NDArray[np.uint8],
        border_width: int = 2,
    ) -> Color:
        """Get color of element border.

        Args:
            region: Image region.
            border_width: Width of border to sample.

        Returns:
            Border color.
        """
        h, w = region.shape[:2]

        # Sample border pixels
        border_pixels = []

        # Top edge
        border_pixels.extend(region[:border_width, :].reshape(-1, 3))
        # Bottom edge
        border_pixels.extend(region[-border_width:, :].reshape(-1, 3))
        # Left edge
        border_pixels.extend(region[:, :border_width].reshape(-1, 3))
        # Right edge
        border_pixels.extend(region[:, -border_width:].reshape(-1, 3))

        if not border_pixels:
            return self._get_average_color(region)

        border_pixels = np.array(border_pixels)
        avg = border_pixels.mean(axis=0).astype(int)
        return Color.from_bgr(tuple(avg))

    async def to_have_color(
        self,
        expected_color: str | Color,
        screenshot: NDArray[np.uint8],
        tolerance: int = 30,
    ) -> AssertionResult:
        """Assert element has specific dominant color.

        Args:
            expected_color: Expected color (hex string or Color).
            screenshot: Screenshot to analyze.
            tolerance: Color distance tolerance.

        Returns:
            Assertion result.
        """
        start_time = time.monotonic()

        if isinstance(expected_color, str):
            expected = Color.from_hex(expected_color)
        else:
            expected = expected_color

        # Find element
        match = await self._locator.find(screenshot)
        if match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return AssertionResult(
                assertion_id="attribute_color",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_color",
                status=AssertionStatus.FAILED,
                message="Element not found",
                expected_value=expected.to_hex(),
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        # Extract region and get dominant color
        region = self._extract_region(screenshot, match.bounds)
        actual = self._get_dominant_color(region)

        # Calculate distance
        distance = expected.distance(actual)
        matches = distance <= tolerance

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        if matches:
            return AssertionResult(
                assertion_id="attribute_color",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_color",
                status=AssertionStatus.PASSED,
                message=f"Color matches: {actual.to_hex()} (distance: {distance:.0f})",
                expected_value=expected.to_hex(),
                actual_value=actual.to_hex(),
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="attribute_color",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_color",
                status=AssertionStatus.FAILED,
                message=f"Color mismatch: expected {expected.to_hex()}, got {actual.to_hex()} (distance: {distance:.0f})",
                expected_value=expected.to_hex(),
                actual_value=actual.to_hex(),
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_have_background_color(
        self,
        expected_color: str | Color,
        screenshot: NDArray[np.uint8],
        tolerance: int = 30,
    ) -> AssertionResult:
        """Assert element has specific background color.

        Samples the interior of the element, avoiding borders.

        Args:
            expected_color: Expected color (hex string or Color).
            screenshot: Screenshot to analyze.
            tolerance: Color distance tolerance.

        Returns:
            Assertion result.
        """
        start_time = time.monotonic()

        if isinstance(expected_color, str):
            expected = Color.from_hex(expected_color)
        else:
            expected = expected_color

        # Find element
        match = await self._locator.find(screenshot)
        if match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return AssertionResult(
                assertion_id="attribute_bg_color",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_background_color",
                status=AssertionStatus.FAILED,
                message="Element not found",
                expected_value=expected.to_hex(),
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        # Extract inner region (avoiding borders)
        region = self._extract_region(screenshot, match.bounds)
        h, w = region.shape[:2]

        # Sample center portion
        margin = min(5, h // 4, w // 4)
        inner = region[margin : h - margin, margin : w - margin]

        if inner.size == 0:
            inner = region

        actual = self._get_average_color(inner)
        distance = expected.distance(actual)
        matches = distance <= tolerance

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        if matches:
            return AssertionResult(
                assertion_id="attribute_bg_color",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_background_color",
                status=AssertionStatus.PASSED,
                message=f"Background color matches: {actual.to_hex()}",
                expected_value=expected.to_hex(),
                actual_value=actual.to_hex(),
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="attribute_bg_color",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_background_color",
                status=AssertionStatus.FAILED,
                message=f"Background color mismatch: expected {expected.to_hex()}, got {actual.to_hex()}",
                expected_value=expected.to_hex(),
                actual_value=actual.to_hex(),
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_have_border_color(
        self,
        expected_color: str | Color,
        screenshot: NDArray[np.uint8],
        tolerance: int = 30,
    ) -> AssertionResult:
        """Assert element has specific border color.

        Args:
            expected_color: Expected color.
            screenshot: Screenshot to analyze.
            tolerance: Color distance tolerance.

        Returns:
            Assertion result.
        """
        start_time = time.monotonic()

        if isinstance(expected_color, str):
            expected = Color.from_hex(expected_color)
        else:
            expected = expected_color

        match = await self._locator.find(screenshot)
        if match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return AssertionResult(
                assertion_id="attribute_border_color",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_border_color",
                status=AssertionStatus.FAILED,
                message="Element not found",
                expected_value=expected.to_hex(),
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        region = self._extract_region(screenshot, match.bounds)
        actual = self._get_border_color(region)
        distance = expected.distance(actual)
        matches = distance <= tolerance

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        if matches:
            return AssertionResult(
                assertion_id="attribute_border_color",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_border_color",
                status=AssertionStatus.PASSED,
                message=f"Border color matches: {actual.to_hex()}",
                expected_value=expected.to_hex(),
                actual_value=actual.to_hex(),
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="attribute_border_color",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_border_color",
                status=AssertionStatus.FAILED,
                message=f"Border color mismatch: expected {expected.to_hex()}, got {actual.to_hex()}",
                expected_value=expected.to_hex(),
                actual_value=actual.to_hex(),
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_have_size(
        self,
        expected_width: int,
        expected_height: int,
        screenshot: NDArray[np.uint8],
        tolerance: int = 5,
    ) -> AssertionResult:
        """Assert element has specific size.

        Args:
            expected_width: Expected width in pixels.
            expected_height: Expected height in pixels.
            screenshot: Screenshot to analyze.
            tolerance: Size tolerance in pixels.

        Returns:
            Assertion result.
        """
        start_time = time.monotonic()

        match = await self._locator.find(screenshot)
        if match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return AssertionResult(
                assertion_id="attribute_size",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_size",
                status=AssertionStatus.FAILED,
                message="Element not found",
                expected_value=f"{expected_width}x{expected_height}",
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        actual_width = match.bounds.width
        actual_height = match.bounds.height

        width_matches = abs(actual_width - expected_width) <= tolerance
        height_matches = abs(actual_height - expected_height) <= tolerance
        matches = width_matches and height_matches

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        if matches:
            return AssertionResult(
                assertion_id="attribute_size",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_size",
                status=AssertionStatus.PASSED,
                message=f"Size matches: {actual_width}x{actual_height}",
                expected_value=f"{expected_width}x{expected_height}",
                actual_value=f"{actual_width}x{actual_height}",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="attribute_size",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_size",
                status=AssertionStatus.FAILED,
                message=f"Size mismatch: expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}",
                expected_value=f"{expected_width}x{expected_height}",
                actual_value=f"{actual_width}x{actual_height}",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_have_width(
        self,
        expected_width: int,
        screenshot: NDArray[np.uint8],
        tolerance: int = 5,
    ) -> AssertionResult:
        """Assert element has specific width.

        Args:
            expected_width: Expected width in pixels.
            screenshot: Screenshot to analyze.
            tolerance: Width tolerance in pixels.

        Returns:
            Assertion result.
        """
        start_time = time.monotonic()

        match = await self._locator.find(screenshot)
        if match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return AssertionResult(
                assertion_id="attribute_width",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_width",
                status=AssertionStatus.FAILED,
                message="Element not found",
                expected_value=expected_width,
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        actual_width = match.bounds.width
        matches = abs(actual_width - expected_width) <= tolerance

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        if matches:
            return AssertionResult(
                assertion_id="attribute_width",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_width",
                status=AssertionStatus.PASSED,
                message=f"Width matches: {actual_width}px",
                expected_value=expected_width,
                actual_value=actual_width,
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="attribute_width",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_width",
                status=AssertionStatus.FAILED,
                message=f"Width mismatch: expected {expected_width}px, got {actual_width}px",
                expected_value=expected_width,
                actual_value=actual_width,
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_have_height(
        self,
        expected_height: int,
        screenshot: NDArray[np.uint8],
        tolerance: int = 5,
    ) -> AssertionResult:
        """Assert element has specific height.

        Args:
            expected_height: Expected height in pixels.
            screenshot: Screenshot to analyze.
            tolerance: Height tolerance in pixels.

        Returns:
            Assertion result.
        """
        start_time = time.monotonic()

        match = await self._locator.find(screenshot)
        if match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return AssertionResult(
                assertion_id="attribute_height",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_height",
                status=AssertionStatus.FAILED,
                message="Element not found",
                expected_value=expected_height,
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        actual_height = match.bounds.height
        matches = abs(actual_height - expected_height) <= tolerance

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        if matches:
            return AssertionResult(
                assertion_id="attribute_height",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_height",
                status=AssertionStatus.PASSED,
                message=f"Height matches: {actual_height}px",
                expected_value=expected_height,
                actual_value=actual_height,
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="attribute_height",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_height",
                status=AssertionStatus.FAILED,
                message=f"Height mismatch: expected {expected_height}px, got {actual_height}px",
                expected_value=expected_height,
                actual_value=actual_height,
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_have_position(
        self,
        expected_x: int,
        expected_y: int,
        screenshot: NDArray[np.uint8],
        tolerance: int = 5,
    ) -> AssertionResult:
        """Assert element is at specific position.

        Args:
            expected_x: Expected X coordinate.
            expected_y: Expected Y coordinate.
            screenshot: Screenshot to analyze.
            tolerance: Position tolerance in pixels.

        Returns:
            Assertion result.
        """
        start_time = time.monotonic()

        match = await self._locator.find(screenshot)
        if match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return AssertionResult(
                assertion_id="attribute_position",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_position",
                status=AssertionStatus.FAILED,
                message="Element not found",
                expected_value=f"({expected_x}, {expected_y})",
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        actual_x = match.bounds.x
        actual_y = match.bounds.y

        x_matches = abs(actual_x - expected_x) <= tolerance
        y_matches = abs(actual_y - expected_y) <= tolerance
        matches = x_matches and y_matches

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        if matches:
            return AssertionResult(
                assertion_id="attribute_position",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_position",
                status=AssertionStatus.PASSED,
                message=f"Position matches: ({actual_x}, {actual_y})",
                expected_value=f"({expected_x}, {expected_y})",
                actual_value=f"({actual_x}, {actual_y})",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="attribute_position",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_have_position",
                status=AssertionStatus.FAILED,
                message=f"Position mismatch: expected ({expected_x}, {expected_y}), got ({actual_x}, {actual_y})",
                expected_value=f"({expected_x}, {expected_y})",
                actual_value=f"({actual_x}, {actual_y})",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_be_within_bounds(
        self,
        container: BoundingBox,
        screenshot: NDArray[np.uint8],
    ) -> AssertionResult:
        """Assert element is completely within container bounds.

        Args:
            container: Container bounding box.
            screenshot: Screenshot to analyze.

        Returns:
            Assertion result.
        """
        start_time = time.monotonic()

        match = await self._locator.find(screenshot)
        if match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return AssertionResult(
                assertion_id="attribute_within_bounds",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_be_within_bounds",
                status=AssertionStatus.FAILED,
                message="Element not found",
                expected_value=f"within ({container.x}, {container.y}, {container.width}, {container.height})",
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        b = match.bounds

        # Check if element is within container
        within = (
            b.x >= container.x
            and b.y >= container.y
            and b.x + b.width <= container.x + container.width
            and b.y + b.height <= container.y + container.height
        )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        if within:
            return AssertionResult(
                assertion_id="attribute_within_bounds",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_be_within_bounds",
                status=AssertionStatus.PASSED,
                message="Element is within bounds",
                expected_value=f"within ({container.x}, {container.y}, {container.width}, {container.height})",
                actual_value=f"at ({b.x}, {b.y}, {b.width}, {b.height})",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="attribute_within_bounds",
                locator_value=self._locator._value,
                assertion_type=AssertionType.ATTRIBUTE,
                assertion_method="to_be_within_bounds",
                status=AssertionStatus.FAILED,
                message="Element extends outside container bounds",
                expected_value=f"within ({container.x}, {container.y}, {container.width}, {container.height})",
                actual_value=f"at ({b.x}, {b.y}, {b.width}, {b.height})",
                matches_found=0,
                duration_ms=elapsed_ms,
            )


__all__ = [
    "AttributeAssertion",
    "Color",
    "Position",
    "Size",
]
