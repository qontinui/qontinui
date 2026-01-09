"""High-level verification DSL for visual assertions.

Provides a fluent, readable API for visual verification:

    # Element verification
    await verify.element("button.png").is_visible()
    await verify.element("submit").is_enabled()
    await verify.text("Welcome").exists()

    # Spatial verification
    await verify.element("cancel").is_left_of("submit")
    await verify.element("header").is_above("content")
    await verify.elements("nav_items").are_aligned(axis="horizontal")

    # Layout verification
    await verify.region(header).has_grid_layout(columns=3)
    await verify.elements("cards").have_consistent_spacing()

    # Text metrics verification
    await verify.text("Title").has_font_size(approximately=24)
    await verify.text_in_region(content).is_left_aligned()

    # Screenshot verification
    await verify.screenshot().matches_baseline("login_page")
    await verify.region(form).is_stable(for_seconds=2)
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.assertions import BoundingBox

if TYPE_CHECKING:
    from qontinui_schemas.testing.environment import GUIEnvironment

    from qontinui.vision.verification.config import VisionConfig
    from qontinui.vision.verification.locators import BaseLocator

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class VerificationResult:
    """Result of a verification check."""

    passed: bool
    message: str
    expected: Any = None
    actual: Any = None
    screenshot_path: Path | None = None
    details: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


class VerificationError(Exception):
    """Raised when a verification fails."""

    def __init__(self, result: VerificationResult) -> None:
        self.result = result
        super().__init__(result.message)


class ElementVerifier:
    """Verifier for a single element.

    Provides fluent API for element assertions:
        await verify.element("button.png").is_visible()
        await verify.element("submit").is_enabled().is_above("footer")
    """

    def __init__(
        self,
        verifier: "Verifier",
        target: "str | Path | BoundingBox | BaseLocator",
    ) -> None:
        self._verifier = verifier
        self._target = target
        self._negated = False
        self._timeout: float | None = None
        self._soft = False

    def not_(self) -> "ElementVerifier":
        """Negate the next assertion."""
        self._negated = True
        return self

    def with_timeout(self, seconds: float) -> "ElementVerifier":
        """Set timeout for the next assertion."""
        self._timeout = seconds
        return self

    def soft(self) -> "ElementVerifier":
        """Make assertion soft (don't raise on failure)."""
        self._soft = True
        return self

    async def is_visible(self, confidence: float = 0.8) -> VerificationResult:
        """Assert element is visible."""
        from qontinui.vision.verification.assertions.visibility import VisibilityAssertion

        screenshot = await self._verifier._get_screenshot()
        locator = self._verifier._resolve_target(self._target)

        assertion = VisibilityAssertion(
            locator=locator,
            config=self._verifier._config,
        )

        timeout_ms = int(self._timeout * 1000) if self._timeout is not None else None
        assertion_result = await assertion.to_be_visible(
            screenshot=screenshot,
            timeout_ms=timeout_ms,
        )

        # Convert AssertionResult to VerificationResult
        result = VerificationResult(
            passed=assertion_result.status.value == "passed",
            message=assertion_result.message,
            expected=assertion_result.expected_value,
            actual=assertion_result.actual_value,
        )

        if self._negated:
            result = VerificationResult(
                passed=not result.passed,
                message=f"Expected element NOT to be visible: {result.message}",
                expected="not visible",
                actual="visible" if not result.passed else "not visible",
            )

        return self._handle_result(result)

    async def is_hidden(self) -> VerificationResult:
        """Assert element is hidden (not visible)."""
        self._negated = not self._negated
        return await self.is_visible()

    async def is_enabled(self) -> VerificationResult:
        """Assert element appears enabled."""
        from qontinui.vision.verification.assertions.state import StateAssertion

        screenshot = await self._verifier._get_screenshot()
        locator = self._verifier._resolve_target(self._target)

        assertion = StateAssertion(
            locator=locator,
            config=self._verifier._config,
            environment=self._verifier._environment,
        )

        assertion_result = await assertion.to_be_enabled(
            screenshot=screenshot,
        )

        # Convert AssertionResult to VerificationResult
        result = VerificationResult(
            passed=assertion_result.status.value == "passed",
            message=assertion_result.message,
            expected=assertion_result.expected_value,
            actual=assertion_result.actual_value,
        )

        if self._negated:
            result = VerificationResult(
                passed=not result.passed,
                message=f"Expected element NOT to be enabled: {result.message}",
                expected="not enabled",
                actual="enabled" if not result.passed else "not enabled",
            )

        return self._handle_result(result)

    async def is_disabled(self) -> VerificationResult:
        """Assert element appears disabled."""
        self._negated = not self._negated
        return await self.is_enabled()

    async def is_above(self, other: "str | Path | BoundingBox") -> VerificationResult:
        """Assert element is above another element."""
        from qontinui.vision.verification.assertions.spatial import SpatialAssertion

        screenshot = await self._verifier._get_screenshot()
        source_locator = self._verifier._resolve_target(self._target)
        target_locator = self._verifier._resolve_target(other)

        assertion = SpatialAssertion(
            locator=source_locator,
            config=self._verifier._config,
        )

        assertion_result = await assertion.to_be_above(
            reference=target_locator,
            screenshot=screenshot,
        )

        # Convert AssertionResult to VerificationResult
        result = VerificationResult(
            passed=assertion_result.status.value == "passed",
            message=assertion_result.message,
            expected=assertion_result.expected_value,
            actual=assertion_result.actual_value,
        )

        if self._negated:
            result = VerificationResult(
                passed=not result.passed,
                message=f"Expected element NOT to be above: {result.message}",
            )

        return self._handle_result(result)

    async def is_below(self, other: "str | Path | BoundingBox") -> VerificationResult:
        """Assert element is below another element."""
        from qontinui.vision.verification.assertions.spatial import SpatialAssertion

        screenshot = await self._verifier._get_screenshot()
        source_locator = self._verifier._resolve_target(self._target)
        target_locator = self._verifier._resolve_target(other)

        assertion = SpatialAssertion(
            locator=source_locator,
            config=self._verifier._config,
        )

        assertion_result = await assertion.to_be_below(
            reference=target_locator,
            screenshot=screenshot,
        )

        # Convert AssertionResult to VerificationResult
        result = VerificationResult(
            passed=assertion_result.status.value == "passed",
            message=assertion_result.message,
            expected=assertion_result.expected_value,
            actual=assertion_result.actual_value,
        )

        if self._negated:
            result = VerificationResult(
                passed=not result.passed,
                message=f"Expected element NOT to be below: {result.message}",
            )

        return self._handle_result(result)

    async def is_left_of(self, other: "str | Path | BoundingBox") -> VerificationResult:
        """Assert element is to the left of another element."""
        from qontinui.vision.verification.assertions.spatial import SpatialAssertion

        screenshot = await self._verifier._get_screenshot()
        source_locator = self._verifier._resolve_target(self._target)
        target_locator = self._verifier._resolve_target(other)

        assertion = SpatialAssertion(
            locator=source_locator,
            config=self._verifier._config,
        )

        assertion_result = await assertion.to_be_left_of(
            reference=target_locator,
            screenshot=screenshot,
        )

        # Convert AssertionResult to VerificationResult
        result = VerificationResult(
            passed=assertion_result.status.value == "passed",
            message=assertion_result.message,
            expected=assertion_result.expected_value,
            actual=assertion_result.actual_value,
        )

        if self._negated:
            result = VerificationResult(
                passed=not result.passed,
                message=f"Expected element NOT to be left of: {result.message}",
            )

        return self._handle_result(result)

    async def is_right_of(self, other: "str | Path | BoundingBox") -> VerificationResult:
        """Assert element is to the right of another element."""
        from qontinui.vision.verification.assertions.spatial import SpatialAssertion

        screenshot = await self._verifier._get_screenshot()
        source_locator = self._verifier._resolve_target(self._target)
        target_locator = self._verifier._resolve_target(other)

        assertion = SpatialAssertion(
            locator=source_locator,
            config=self._verifier._config,
        )

        assertion_result = await assertion.to_be_right_of(
            reference=target_locator,
            screenshot=screenshot,
        )

        # Convert AssertionResult to VerificationResult
        result = VerificationResult(
            passed=assertion_result.status.value == "passed",
            message=assertion_result.message,
            expected=assertion_result.expected_value,
            actual=assertion_result.actual_value,
        )

        if self._negated:
            result = VerificationResult(
                passed=not result.passed,
                message=f"Expected element NOT to be right of: {result.message}",
            )

        return self._handle_result(result)

    async def has_text(self, expected: str, exact: bool = False) -> VerificationResult:
        """Assert element contains or matches text."""
        from qontinui.vision.verification.assertions.text import TextAssertion

        screenshot = await self._verifier._get_screenshot()
        locator = self._verifier._resolve_target(self._target)

        assertion = TextAssertion(
            locator=locator,
            config=self._verifier._config,
            environment=self._verifier._environment,
        )

        timeout_ms = int(self._timeout * 1000) if self._timeout is not None else None

        if exact:
            assertion_result = await assertion.to_have_text(
                expected_text=expected,
                screenshot=screenshot,
                timeout_ms=timeout_ms,
                exact=True,
            )
        else:
            assertion_result = await assertion.to_contain_text(
                expected_text=expected,
                screenshot=screenshot,
                timeout_ms=timeout_ms,
            )

        # Convert AssertionResult to VerificationResult
        result = VerificationResult(
            passed=assertion_result.status.value == "passed",
            message=assertion_result.message,
            expected=assertion_result.expected_value,
            actual=assertion_result.actual_value,
        )

        if self._negated:
            result = VerificationResult(
                passed=not result.passed,
                message=f"Expected element NOT to have text '{expected}': {result.message}",
                expected=f"not '{expected}'",
                actual=result.actual,
            )

        return self._handle_result(result)

    async def has_color(
        self,
        expected: tuple[int, int, int] | str,
        tolerance: int = 30,
    ) -> VerificationResult:
        """Assert element has expected color."""
        from qontinui.vision.verification.assertions.attributes import (
            AttributeAssertion,
            Color,
        )

        screenshot = await self._verifier._get_screenshot()
        locator = self._verifier._resolve_target(self._target)

        assertion = AttributeAssertion(
            locator=locator,
            config=self._verifier._config,
            environment=self._verifier._environment,
        )

        if isinstance(expected, str):
            color = Color.from_hex(expected)
        else:
            color = Color(r=expected[0], g=expected[1], b=expected[2])

        assertion_result = await assertion.to_have_color(
            expected_color=color,
            screenshot=screenshot,
            tolerance=tolerance,
        )

        # Convert AssertionResult to VerificationResult
        result = VerificationResult(
            passed=assertion_result.status.value == "passed",
            message=assertion_result.message,
            expected=assertion_result.expected_value,
            actual=assertion_result.actual_value,
        )

        if self._negated:
            result = VerificationResult(
                passed=not result.passed,
                message=f"Expected element NOT to have color {color}: {result.message}",
            )

        return self._handle_result(result)

    def _handle_result(self, result: VerificationResult) -> VerificationResult:
        """Handle verification result."""
        # Reset state
        self._negated = False
        self._timeout = None

        if not result.passed and not self._soft:
            raise VerificationError(result)

        self._soft = False
        return result


class TextVerifier:
    """Verifier for text content.

    Provides fluent API for text assertions:
        await verify.text("Welcome").exists()
        await verify.text("Error").not_().exists()
    """

    def __init__(self, verifier: "Verifier", text: str) -> None:
        self._verifier = verifier
        self._text = text
        self._negated = False
        self._timeout: float | None = None
        self._soft = False

    def not_(self) -> "TextVerifier":
        """Negate the next assertion."""
        self._negated = True
        return self

    def with_timeout(self, seconds: float) -> "TextVerifier":
        """Set timeout for the next assertion."""
        self._timeout = seconds
        return self

    def soft(self) -> "TextVerifier":
        """Make assertion soft (don't raise on failure)."""
        self._soft = True
        return self

    async def exists(self, confidence: float = 0.8) -> VerificationResult:
        """Assert text exists on screen."""
        from qontinui.vision.verification.assertions.text import TextAssertion

        screenshot = await self._verifier._get_screenshot()

        # TextAssertion without a locator will search the entire screenshot
        assertion = TextAssertion(
            locator=None,
            config=self._verifier._config,
            environment=self._verifier._environment,
        )

        timeout_ms = int(self._timeout * 1000) if self._timeout is not None else None

        # Use to_contain_text to check if text exists anywhere on screen
        assertion_result = await assertion.to_contain_text(
            expected_text=self._text,
            screenshot=screenshot,
            timeout_ms=timeout_ms,
        )

        # Convert AssertionResult to VerificationResult
        result = VerificationResult(
            passed=assertion_result.status.value == "passed",
            message=assertion_result.message,
            expected=assertion_result.expected_value,
            actual=assertion_result.actual_value,
        )

        if self._negated:
            result = VerificationResult(
                passed=not result.passed,
                message=f"Expected text '{self._text}' NOT to exist: {result.message}",
                expected="not present",
                actual="present" if not result.passed else "not present",
            )

        return self._handle_result(result)

    async def has_font_size(
        self,
        size: int | None = None,
        approximately: int | None = None,
        tolerance: int = 2,
    ) -> VerificationResult:
        """Assert text has expected font size."""
        from qontinui.vision.verification.analysis.text_metrics import TextMetricsAnalyzer
        from qontinui.vision.verification.locators import TextLocator

        screenshot = await self._verifier._get_screenshot()
        locator = TextLocator(
            text=self._text,
            config=self._verifier._config,
            environment=self._verifier._environment,
        )

        # Find text location
        match = await locator.find(screenshot)
        if not match:
            result = VerificationResult(
                passed=False,
                message=f"Text '{self._text}' not found",
                expected=size or approximately,
                actual=None,
            )
            return self._handle_result(result)

        # Analyze text metrics
        analyzer = TextMetricsAnalyzer(
            config=self._verifier._config,
            environment=self._verifier._environment,
        )

        metrics = await analyzer.analyze_region(screenshot, match.bounds)
        actual_size = metrics.average_font_size

        expected = size if size is not None else approximately
        if expected is None:
            result = VerificationResult(
                passed=False,
                message="No expected font size provided",
            )
            return self._handle_result(result)

        passed = abs(actual_size - expected) <= tolerance

        result = VerificationResult(
            passed=passed,
            message=f"Font size: expected ~{expected}px, got {actual_size:.1f}px",
            expected=expected,
            actual=actual_size,
        )

        if self._negated:
            result = VerificationResult(
                passed=not result.passed,
                message=f"Expected font size NOT to be ~{expected}px: {result.message}",
                expected=f"not ~{expected}",
                actual=actual_size,
            )

        return self._handle_result(result)

    def _handle_result(self, result: VerificationResult) -> VerificationResult:
        """Handle verification result."""
        self._negated = False
        self._timeout = None

        if not result.passed and not self._soft:
            raise VerificationError(result)

        self._soft = False
        return result


class RegionVerifier:
    """Verifier for a screen region.

    Provides fluent API for region assertions:
        await verify.region(header).has_grid_layout(columns=3)
        await verify.region(content).is_stable(for_seconds=2)
    """

    def __init__(self, verifier: "Verifier", region: BoundingBox) -> None:
        self._verifier = verifier
        self._region = region
        self._negated = False
        self._timeout: float | None = None
        self._soft = False

    def not_(self) -> "RegionVerifier":
        """Negate the next assertion."""
        self._negated = True
        return self

    def soft(self) -> "RegionVerifier":
        """Make assertion soft (don't raise on failure)."""
        self._soft = True
        return self

    async def is_stable(self, for_seconds: float = 1.0) -> VerificationResult:
        """Assert region is visually stable (no animation)."""
        from qontinui.vision.verification.assertions.animation import StabilityDetector

        detector = StabilityDetector(
            config=self._verifier._config,
        )

        stability_duration_ms = int(for_seconds * 1000)
        stability = await detector.wait_for_stability(
            screenshot_callback=self._verifier._get_screenshot,
            region=self._region,
            stability_duration_ms=stability_duration_ms,
        )

        passed = stability.is_stable

        result = VerificationResult(
            passed=passed,
            message=f"Region stability: max change {stability.max_change_detected:.2%}",
            expected="stable",
            actual="stable" if passed else "unstable",
            details={
                "max_change_detected": stability.max_change_detected,
                "stability_duration_ms": stability.stability_duration_ms,
            },
        )

        if self._negated:
            result = VerificationResult(
                passed=not result.passed,
                message=f"Expected region NOT to be stable: {result.message}",
            )

        return self._handle_result(result)

    async def has_grid_layout(
        self,
        columns: int | None = None,
        rows: int | None = None,
        tolerance: int = 1,
    ) -> VerificationResult:
        """Assert region contains a grid layout."""
        import cv2

        from qontinui.vision.verification.analysis.layout import LayoutAnalyzer

        analyzer = LayoutAnalyzer(
            config=self._verifier._config,
            environment=self._verifier._environment,
        )

        screenshot = await self._verifier._get_screenshot()

        # Extract the region from the screenshot
        region_img = screenshot[
            self._region.y : self._region.y + self._region.height,
            self._region.x : self._region.x + self._region.width,
        ]

        # Detect elements in the region using contour detection
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert contours to BoundingBox list (filter small noise)
        elements = []
        min_area = 100  # Minimum area to be considered an element
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= min_area:
                elements.append(
                    BoundingBox(
                        x=self._region.x + x,
                        y=self._region.y + y,
                        width=w,
                        height=h,
                    )
                )

        if len(elements) < 2:
            result = VerificationResult(
                passed=False,
                message=f"Not enough elements detected for grid analysis ({len(elements)} found)",
                expected=f"{columns}x{rows}" if columns and rows else "grid",
                actual="insufficient elements",
            )
            return self._handle_result(result)

        # Analyze the layout using the detected elements
        structure = analyzer.analyze_layout(elements)

        grid = structure.grid
        if grid is None:
            result = VerificationResult(
                passed=False,
                message="No grid layout detected",
                expected=f"{columns}x{rows}" if columns and rows else "grid",
                actual="no grid",
            )
            return self._handle_result(result)

        passed = True
        if columns is not None:
            passed = passed and abs(grid.columns - columns) <= tolerance
        if rows is not None:
            passed = passed and abs(grid.rows - rows) <= tolerance

        result = VerificationResult(
            passed=passed,
            message=f"Grid layout: {grid.columns} columns x {grid.rows} rows",
            expected=f"{columns}x{rows}" if columns and rows else "grid",
            actual=f"{grid.columns}x{grid.rows}",
            details={"grid": grid},
        )

        if self._negated:
            result = VerificationResult(
                passed=not result.passed,
                message=f"Expected region NOT to have grid layout: {result.message}",
            )

        return self._handle_result(result)

    async def text_is_aligned(self, alignment: str = "left") -> VerificationResult:
        """Assert text in region has expected alignment."""
        from qontinui.vision.verification.analysis.text_metrics import (
            TextAlignment,
            TextMetricsAnalyzer,
        )

        analyzer = TextMetricsAnalyzer(
            config=self._verifier._config,
            environment=self._verifier._environment,
        )

        screenshot = await self._verifier._get_screenshot()
        metrics = await analyzer.analyze_region(screenshot, self._region)

        expected_alignment = TextAlignment(alignment.lower())
        actual_alignment = metrics.alignment

        passed = actual_alignment == expected_alignment

        result = VerificationResult(
            passed=passed,
            message=f"Text alignment: expected {expected_alignment.value}, got {actual_alignment.value}",
            expected=expected_alignment.value,
            actual=actual_alignment.value,
        )

        if self._negated:
            result = VerificationResult(
                passed=not result.passed,
                message=f"Expected text NOT to be {alignment} aligned: {result.message}",
            )

        return self._handle_result(result)

    def _handle_result(self, result: VerificationResult) -> VerificationResult:
        """Handle verification result."""
        self._negated = False
        self._timeout = None

        if not result.passed and not self._soft:
            raise VerificationError(result)

        self._soft = False
        return result


class ScreenshotVerifier:
    """Verifier for screenshot comparison.

    Provides fluent API for visual regression:
        await verify.screenshot().matches_baseline("login_page")
    """

    def __init__(self, verifier: "Verifier") -> None:
        self._verifier = verifier
        self._region: BoundingBox | None = None
        self._soft = False

    def in_region(self, region: BoundingBox) -> "ScreenshotVerifier":
        """Limit comparison to a region."""
        self._region = region
        return self

    def soft(self) -> "ScreenshotVerifier":
        """Make assertion soft (don't raise on failure)."""
        self._soft = True
        return self

    async def matches_baseline(
        self,
        name: str,
        threshold: float = 0.95,
        method: str = "ssim",
    ) -> VerificationResult:
        """Assert screenshot matches baseline."""
        from qontinui.vision.verification.assertions.screenshot import (
            ScreenshotAssertion,
        )

        assertion = ScreenshotAssertion(
            config=self._verifier._config,
            environment=self._verifier._environment,
        )

        screenshot = await self._verifier._get_screenshot()

        # Crop to region if specified
        actual = screenshot
        if self._region is not None:
            actual = screenshot[
                self._region.y : self._region.y + self._region.height,
                self._region.x : self._region.x + self._region.width,
            ]

        assertion_result = await assertion.to_match_screenshot(
            actual=actual,
            baseline_name=name,
            threshold=threshold,
            method=method,
        )

        # Convert AssertionResult to VerificationResult
        result = VerificationResult(
            passed=assertion_result.status.value == "passed",
            message=assertion_result.message,
            expected=assertion_result.expected_value,
            actual=assertion_result.actual_value,
        )

        return self._handle_result(result)

    def _handle_result(self, result: VerificationResult) -> VerificationResult:
        """Handle verification result."""
        if not result.passed and not self._soft:
            raise VerificationError(result)

        self._soft = False
        return result


class ElementsVerifier:
    """Verifier for multiple elements.

    Provides fluent API for multi-element assertions:
        await verify.elements("nav_items").are_aligned(axis="horizontal")
        await verify.elements("cards").have_consistent_spacing()
    """

    def __init__(
        self,
        verifier: "Verifier",
        targets: list[str | Path | BoundingBox],
    ) -> None:
        self._verifier = verifier
        self._targets = targets
        self._negated = False
        self._soft = False

    def not_(self) -> "ElementsVerifier":
        """Negate the next assertion."""
        self._negated = True
        return self

    def soft(self) -> "ElementsVerifier":
        """Make assertion soft (don't raise on failure)."""
        self._soft = True
        return self

    async def are_aligned(
        self,
        axis: str = "horizontal",
        tolerance: int = 5,
    ) -> VerificationResult:
        """Assert elements are aligned along an axis."""
        from qontinui.vision.verification.analysis.layout import LayoutAnalyzer

        screenshot = await self._verifier._get_screenshot()

        # Find all element bounds
        bounds_list = []
        for target in self._targets:
            locator = self._verifier._resolve_target(target)
            match = await locator.find(screenshot)
            if match:
                bounds_list.append(match.bounds)

        if len(bounds_list) < 2:
            result = VerificationResult(
                passed=False,
                message=f"Need at least 2 elements for alignment check, found {len(bounds_list)}",
            )
            return self._handle_result(result)

        _analyzer = LayoutAnalyzer(  # noqa: F841 - prepared for future use
            config=self._verifier._config,
            environment=self._verifier._environment,
        )

        # Check alignment
        if axis == "horizontal":
            # Check if Y centers are aligned
            centers = [b.y + b.height / 2 for b in bounds_list]
            variance = max(centers) - min(centers)
            passed = variance <= tolerance
        elif axis == "vertical":
            # Check if X centers are aligned
            centers = [b.x + b.width / 2 for b in bounds_list]
            variance = max(centers) - min(centers)
            passed = variance <= tolerance
        else:
            result = VerificationResult(
                passed=False,
                message=f"Unknown alignment axis: {axis}",
            )
            return self._handle_result(result)

        result = VerificationResult(
            passed=passed,
            message=f"Elements {axis} alignment variance: {variance:.1f}px (tolerance: {tolerance}px)",
            expected=f"aligned within {tolerance}px",
            actual=f"{variance:.1f}px variance",
        )

        if self._negated:
            result = VerificationResult(
                passed=not result.passed,
                message=f"Expected elements NOT to be {axis}ly aligned: {result.message}",
            )

        return self._handle_result(result)

    async def have_consistent_spacing(
        self,
        tolerance: float = 0.2,
    ) -> VerificationResult:
        """Assert elements have consistent spacing between them."""
        screenshot = await self._verifier._get_screenshot()

        # Find all element bounds
        bounds_list = []
        for target in self._targets:
            locator = self._verifier._resolve_target(target)
            match = await locator.find(screenshot)
            if match:
                bounds_list.append(match.bounds)

        if len(bounds_list) < 3:
            result = VerificationResult(
                passed=False,
                message=f"Need at least 3 elements for spacing check, found {len(bounds_list)}",
            )
            return self._handle_result(result)

        # Sort by position (assume primarily horizontal or vertical arrangement)
        # Try horizontal first
        bounds_list.sort(key=lambda b: b.x)

        # Calculate spacings
        spacings = []
        for i in range(1, len(bounds_list)):
            prev = bounds_list[i - 1]
            curr = bounds_list[i]
            spacing = curr.x - (prev.x + prev.width)
            spacings.append(spacing)

        avg_spacing = sum(spacings) / len(spacings)
        max_deviation = max(abs(s - avg_spacing) for s in spacings)
        relative_deviation = max_deviation / avg_spacing if avg_spacing > 0 else float("inf")

        passed = relative_deviation <= tolerance

        result = VerificationResult(
            passed=passed,
            message=f"Spacing consistency: avg={avg_spacing:.1f}px, max deviation={relative_deviation:.1%}",
            expected=f"deviation <= {tolerance:.0%}",
            actual=f"{relative_deviation:.1%} deviation",
            details={"spacings": spacings, "average": avg_spacing},
        )

        if self._negated:
            result = VerificationResult(
                passed=not result.passed,
                message=f"Expected elements NOT to have consistent spacing: {result.message}",
            )

        return self._handle_result(result)

    async def have_count(self, expected: int) -> VerificationResult:
        """Assert expected number of elements are found."""
        screenshot = await self._verifier._get_screenshot()

        # Count found elements
        found_count = 0
        for target in self._targets:
            locator = self._verifier._resolve_target(target)
            match = await locator.find(screenshot)
            if match:
                found_count += 1

        passed = found_count == expected

        result = VerificationResult(
            passed=passed,
            message=f"Element count: expected {expected}, found {found_count}",
            expected=expected,
            actual=found_count,
        )

        if self._negated:
            result = VerificationResult(
                passed=not result.passed,
                message=f"Expected NOT to find {expected} elements: {result.message}",
            )

        return self._handle_result(result)

    def _handle_result(self, result: VerificationResult) -> VerificationResult:
        """Handle verification result."""
        self._negated = False

        if not result.passed and not self._soft:
            raise VerificationError(result)

        self._soft = False
        return result


class Verifier:
    """Main verifier class providing the DSL entry points.

    Usage:
        verify = Verifier(screenshot_callback=capture_screen)

        # Element verification
        await verify.element("button.png").is_visible()
        await verify.element("submit").is_enabled()

        # Text verification
        await verify.text("Welcome").exists()
        await verify.text("Error").not_().exists()

        # Region verification
        await verify.region(header_bounds).is_stable(for_seconds=2)

        # Screenshot verification
        await verify.screenshot().matches_baseline("login_page")

        # Multi-element verification
        await verify.elements(["btn1.png", "btn2.png"]).are_aligned()
    """

    def __init__(
        self,
        screenshot_callback: (
            Callable[[], "NDArray[np.uint8]"]
            | Callable[[], "asyncio.Future[NDArray[np.uint8]]"]
            | None
        ) = None,
        config: "VisionConfig | None" = None,
        environment: "GUIEnvironment | None" = None,
    ) -> None:
        """Initialize verifier.

        Args:
            screenshot_callback: Async or sync function that returns current screenshot.
            config: Vision configuration.
            environment: GUI environment.
        """
        self._screenshot_callback = screenshot_callback
        self._config = config
        self._environment = environment
        self._cached_screenshot: NDArray[np.uint8] | None = None

    def element(self, target: "str | Path | BoundingBox | BaseLocator") -> ElementVerifier:
        """Create element verifier.

        Args:
            target: Element identifier - image path, text, bounds, or locator.
        """
        return ElementVerifier(self, target)

    def text(self, text: str) -> TextVerifier:
        """Create text verifier.

        Args:
            text: Text to verify.
        """
        return TextVerifier(self, text)

    def region(self, bounds: BoundingBox) -> RegionVerifier:
        """Create region verifier.

        Args:
            bounds: Region bounds to verify.
        """
        return RegionVerifier(self, bounds)

    def screenshot(self) -> ScreenshotVerifier:
        """Create screenshot verifier."""
        return ScreenshotVerifier(self)

    def elements(
        self,
        targets: list[str | Path | BoundingBox],
    ) -> ElementsVerifier:
        """Create multi-element verifier.

        Args:
            targets: List of element identifiers.
        """
        return ElementsVerifier(self, targets)

    async def _get_screenshot(self) -> NDArray[np.uint8]:
        """Get current screenshot."""
        if self._screenshot_callback is None:
            if self._cached_screenshot is not None:
                return self._cached_screenshot
            raise ValueError("No screenshot callback provided and no cached screenshot")

        result = self._screenshot_callback()
        if asyncio.iscoroutine(result):
            return await result
        # When not a coroutine, result is the NDArray directly
        return result  # type: ignore[return-value]

    def set_screenshot(self, screenshot: NDArray[np.uint8]) -> None:
        """Set a screenshot to use instead of callback.

        Args:
            screenshot: Screenshot to use.
        """
        self._cached_screenshot = screenshot

    def _resolve_target(
        self,
        target: "str | Path | BoundingBox | BaseLocator",
    ) -> "BaseLocator":
        """Resolve target to a locator."""
        from qontinui.vision.verification.locators import (
            ImageLocator,
            RegionLocator,
            TextLocator,
        )

        if hasattr(target, "find"):
            # Already a locator
            return target  # type: ignore

        if isinstance(target, BoundingBox):
            return RegionLocator.from_bounds(
                bounds=target,
                config=self._config,
            )

        if isinstance(target, Path) or (
            isinstance(target, str)
            and (
                target.endswith(".png")
                or target.endswith(".jpg")
                or "/" in target
                or "\\" in target
            )
        ):
            return ImageLocator(
                image_path=Path(target) if isinstance(target, str) else target,
                config=self._config,
                environment=self._environment,
            )

        # Assume text
        return TextLocator(
            text=str(target),
            config=self._config,
            environment=self._environment,
        )


# Convenience function
def create_verifier(
    screenshot_callback: (
        Callable[[], NDArray[np.uint8]] | Callable[[], asyncio.Future[NDArray[np.uint8]]] | None
    ) = None,
    config: "VisionConfig | None" = None,
    environment: "GUIEnvironment | None" = None,
) -> Verifier:
    """Create a verifier instance.

    Args:
        screenshot_callback: Async or sync function that returns current screenshot.
        config: Vision configuration.
        environment: GUI environment.

    Returns:
        Configured Verifier instance.
    """
    return Verifier(
        screenshot_callback=screenshot_callback,
        config=config,
        environment=environment,
    )


__all__ = [
    "create_verifier",
    "ElementsVerifier",
    "ElementVerifier",
    "RegionVerifier",
    "ScreenshotVerifier",
    "TextVerifier",
    "VerificationError",
    "VerificationResult",
    "Verifier",
]
