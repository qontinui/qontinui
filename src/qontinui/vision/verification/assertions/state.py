"""State assertions with environment integration.

Provides assertions for element visual states (enabled, disabled,
focused, checked, etc.) with support for learned visual signatures
from the GUI environment discovery.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.common import utc_now
from qontinui_schemas.testing.assertions import (
    AssertionResult,
    AssertionStatus,
    BoundingBox,
)

if TYPE_CHECKING:
    from qontinui_schemas.testing.environment import ElementState as SchemaElementState
    from qontinui_schemas.testing.environment import (
        GUIEnvironment,
        VisualStates,
    )

    from qontinui.vision.verification.config import VisionConfig
    from qontinui.vision.verification.locators.base import BaseLocator

logger = logging.getLogger(__name__)


class ElementState(str, Enum):
    """Element visual states."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    FOCUSED = "focused"
    SELECTED = "selected"
    CHECKED = "checked"
    UNCHECKED = "unchecked"
    EXPANDED = "expanded"
    COLLAPSED = "collapsed"
    PRESSED = "pressed"
    HOVERED = "hovered"
    ACTIVE = "active"
    INACTIVE = "inactive"


@dataclass
class StateDetectionResult:
    """Result of state detection."""

    detected_state: ElementState | None
    confidence: float
    method: str  # 'environment', 'heuristic', 'template'
    details: dict


class StateDetector:
    """Detects element visual states.

    Integrates with discovered GUI environment for accurate
    state detection based on learned visual signatures.

    Usage:
        detector = StateDetector(config, environment)

        # Detect state
        result = detector.detect_state(element_region, screenshot)

        # Check specific state
        is_enabled = detector.is_enabled(element_region, screenshot)
    """

    def __init__(
        self,
        config: "VisionConfig | None" = None,
        environment: "GUIEnvironment | None" = None,
    ) -> None:
        """Initialize state detector.

        Args:
            config: Vision configuration.
            environment: GUI environment with learned states.
        """
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

    def _get_visual_states(self) -> list["SchemaElementState"]:
        """Get visual states from environment.

        Returns:
            List of learned visual states (flattened from all element types).
        """
        if self._environment is None:
            return []
        visual_states: VisualStates | None = self._environment.visual_states
        if visual_states is None:
            return []
        # Flatten all element states from all element types
        result: list[SchemaElementState] = []
        for element_type_states in visual_states.element_states.values():
            result.extend(element_type_states.states.values())
        return result

    def _match_visual_signature(
        self,
        region: NDArray[np.uint8],
        state: "SchemaElementState",
    ) -> float:
        """Match region against a visual state signature.

        Args:
            region: Element region.
            state: Visual state to match against.

        Returns:
            Match confidence (0.0-1.0).
        """
        # Check color match if defined via visual_signature
        confidence = 0.0
        checks = 0
        sig = state.visual_signature

        # Check color profile if defined (background is a hex string like "#RRGGBB")
        if sig and sig.color_profile and sig.color_profile.background:
            avg_color = cv2.mean(region)[:3]  # BGR format
            bg_hex = sig.color_profile.background
            # Parse hex color to RGB
            r = int(bg_hex[1:3], 16)
            g = int(bg_hex[3:5], 16)
            b = int(bg_hex[5:7], 16)

            # Calculate color distance (avg_color is BGR)
            distance = np.sqrt(
                (avg_color[2] - r) ** 2 + (avg_color[1] - g) ** 2 + (avg_color[0] - b) ** 2
            )

            # Normalize to 0-1 (max distance is ~441)
            color_match = 1.0 - min(float(distance) / 200, 1.0)
            confidence += color_match
            checks += 1

        # Check saturation if relevant (state_type is ElementStateType enum)
        state_type_value = state.state_type.value if state.state_type else ""
        if state_type_value in ["disabled", "enabled"]:
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1].mean()

            if state_type_value == "disabled":
                # Disabled typically has low saturation
                sat_match = 1.0 - min(saturation / 100, 1.0)
            else:
                # Enabled typically has normal/high saturation
                sat_match = min(saturation / 100, 1.0)

            confidence += sat_match
            checks += 1

        # Check brightness
        if state_type_value in ["focused", "selected", "hovered"]:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            brightness = gray.mean()

            # These states often have increased brightness
            bright_match = min(brightness / 200, 1.0)
            confidence += bright_match
            checks += 1

        if checks > 0:
            return confidence / checks

        return 0.5  # Default neutral confidence

    def detect_state(
        self,
        bounds: BoundingBox,
        screenshot: NDArray[np.uint8],
    ) -> StateDetectionResult:
        """Detect element's visual state.

        Args:
            bounds: Element bounding box.
            screenshot: Screenshot to analyze.

        Returns:
            State detection result.
        """
        region = self._extract_region(screenshot, bounds)

        if region.size == 0:
            return StateDetectionResult(
                detected_state=None,
                confidence=0.0,
                method="none",
                details={"error": "Empty region"},
            )

        # Try environment-based detection first
        visual_states = self._get_visual_states()
        if visual_states:
            best_match: ElementState | None = None
            best_confidence = 0.0

            for vs in visual_states:
                confidence = self._match_visual_signature(region, vs)
                if confidence > best_confidence:
                    best_confidence = confidence
                    try:
                        # vs.state_type is ElementStateType enum, get its value
                        state_value = vs.state_type.value if vs.state_type else None
                        if state_value:
                            best_match = ElementState(state_value)
                    except ValueError:
                        pass

            if best_match and best_confidence > 0.6:
                return StateDetectionResult(
                    detected_state=best_match,
                    confidence=best_confidence,
                    method="environment",
                    details={"matched_signatures": len(visual_states)},
                )

        # Fall back to heuristic detection
        return self._detect_state_heuristic(region)

    def _detect_state_heuristic(
        self,
        region: NDArray[np.uint8],
    ) -> StateDetectionResult:
        """Detect state using visual heuristics.

        Args:
            region: Element region.

        Returns:
            State detection result.
        """
        # Convert to HSV for analysis
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].mean()
        brightness = hsv[:, :, 2].mean()

        # Analyze edges for focus ring detection
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = edges.sum() / (region.shape[0] * region.shape[1] * 255)

        details = {
            "saturation": float(saturation),
            "brightness": float(brightness),
            "edge_ratio": float(edge_ratio),
        }

        # Determine most likely state
        if saturation < 30 and brightness < 150:
            return StateDetectionResult(
                detected_state=ElementState.DISABLED,
                confidence=0.7,
                method="heuristic",
                details=details,
            )

        if edge_ratio > 0.15:
            return StateDetectionResult(
                detected_state=ElementState.FOCUSED,
                confidence=0.6,
                method="heuristic",
                details=details,
            )

        if brightness > 200:
            return StateDetectionResult(
                detected_state=ElementState.HOVERED,
                confidence=0.5,
                method="heuristic",
                details=details,
            )

        return StateDetectionResult(
            detected_state=ElementState.ENABLED,
            confidence=0.6,
            method="heuristic",
            details=details,
        )

    def is_enabled(
        self,
        bounds: BoundingBox,
        screenshot: NDArray[np.uint8],
    ) -> tuple[bool, float]:
        """Check if element appears enabled.

        Args:
            bounds: Element bounding box.
            screenshot: Screenshot to analyze.

        Returns:
            Tuple of (is_enabled, confidence).
        """
        result = self.detect_state(bounds, screenshot)

        if result.detected_state == ElementState.ENABLED:
            return True, result.confidence
        elif result.detected_state == ElementState.DISABLED:
            return False, result.confidence

        # Use saturation heuristic
        region = self._extract_region(screenshot, bounds)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].mean()

        is_enabled = saturation >= 30
        confidence = 0.6 if is_enabled else 0.7

        return is_enabled, confidence

    def is_disabled(
        self,
        bounds: BoundingBox,
        screenshot: NDArray[np.uint8],
    ) -> tuple[bool, float]:
        """Check if element appears disabled.

        Args:
            bounds: Element bounding box.
            screenshot: Screenshot to analyze.

        Returns:
            Tuple of (is_disabled, confidence).
        """
        is_enabled, confidence = self.is_enabled(bounds, screenshot)
        return not is_enabled, confidence

    def is_focused(
        self,
        bounds: BoundingBox,
        screenshot: NDArray[np.uint8],
    ) -> tuple[bool, float]:
        """Check if element appears focused.

        Args:
            bounds: Element bounding box.
            screenshot: Screenshot to analyze.

        Returns:
            Tuple of (is_focused, confidence).
        """
        region = self._extract_region(screenshot, bounds)

        # Check for focus ring (bright border)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = edges.sum() / (region.shape[0] * region.shape[1] * 255)

        is_focused = edge_ratio > 0.1
        confidence = min(edge_ratio * 5, 0.9) if is_focused else 0.6

        return is_focused, confidence

    def is_checked(
        self,
        bounds: BoundingBox,
        screenshot: NDArray[np.uint8],
    ) -> tuple[bool, float]:
        """Check if checkbox/toggle appears checked.

        Args:
            bounds: Element bounding box.
            screenshot: Screenshot to analyze.

        Returns:
            Tuple of (is_checked, confidence).
        """
        region = self._extract_region(screenshot, bounds)

        # Check for interior fill (checkmark or filled state)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Look for significant contrast (checkmark)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Calculate fill ratio
        total_pixels = thresh.size
        dark_pixels = np.count_nonzero(thresh == 0)
        light_pixels = np.count_nonzero(thresh == 255)

        # Check color saturation (filled usually has color)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].mean()

        # Multiple indicators of "checked"
        has_contrast = abs(dark_pixels - light_pixels) / total_pixels > 0.1
        has_saturation = saturation > 50

        is_checked = has_contrast or has_saturation
        confidence = 0.7 if is_checked else 0.6

        return is_checked, confidence


class StateAssertion:
    """Assertions for element visual states.

    Integrates with GUI environment for accurate state detection.

    Usage:
        assertion = StateAssertion(locator, config, environment)

        # State assertions
        result = await assertion.to_be_enabled(screenshot)
        result = await assertion.to_be_disabled(screenshot)
        result = await assertion.to_be_focused(screenshot)
        result = await assertion.to_be_checked(screenshot)
    """

    def __init__(
        self,
        locator: "BaseLocator",
        config: "VisionConfig | None" = None,
        environment: "GUIEnvironment | None" = None,
    ) -> None:
        """Initialize state assertion.

        Args:
            locator: Locator for finding the element.
            config: Vision configuration.
            environment: GUI environment with learned states.
        """
        self._locator = locator
        self._config = config
        self._environment = environment
        self._detector = StateDetector(config, environment)

    async def to_be_enabled(
        self,
        screenshot: NDArray[np.uint8],
    ) -> AssertionResult:
        """Assert element appears enabled.

        Args:
            screenshot: Screenshot to analyze.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        start_time = time.monotonic()

        match = await self._locator.find(screenshot)
        if match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            completed_at = utc_now()
            return AssertionResult(
                assertion_id="state_enabled",
                assertion_method="to_be_enabled",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Element not found",
                expected_value="enabled",
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        is_enabled, confidence = self._detector.is_enabled(match.bounds, screenshot)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        completed_at = utc_now()

        if is_enabled:
            return AssertionResult(
                assertion_id="state_enabled",
                assertion_method="to_be_enabled",
                status=AssertionStatus.PASSED,
                started_at=started_at,
                completed_at=completed_at,
                expected_value="enabled",
                actual_value="enabled",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="state_enabled",
                assertion_method="to_be_enabled",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message=f"Element appears disabled (confidence: {confidence:.0%})",
                expected_value="enabled",
                actual_value="disabled",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_be_disabled(
        self,
        screenshot: NDArray[np.uint8],
    ) -> AssertionResult:
        """Assert element appears disabled.

        Args:
            screenshot: Screenshot to analyze.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        start_time = time.monotonic()

        match = await self._locator.find(screenshot)
        if match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            completed_at = utc_now()
            return AssertionResult(
                assertion_id="state_disabled",
                assertion_method="to_be_disabled",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Element not found",
                expected_value="disabled",
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        is_disabled, confidence = self._detector.is_disabled(match.bounds, screenshot)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        completed_at = utc_now()

        if is_disabled:
            return AssertionResult(
                assertion_id="state_disabled",
                assertion_method="to_be_disabled",
                status=AssertionStatus.PASSED,
                started_at=started_at,
                completed_at=completed_at,
                expected_value="disabled",
                actual_value="disabled",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="state_disabled",
                assertion_method="to_be_disabled",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message=f"Element appears enabled (confidence: {confidence:.0%})",
                expected_value="disabled",
                actual_value="enabled",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_be_focused(
        self,
        screenshot: NDArray[np.uint8],
    ) -> AssertionResult:
        """Assert element appears focused.

        Args:
            screenshot: Screenshot to analyze.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        start_time = time.monotonic()

        match = await self._locator.find(screenshot)
        if match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            completed_at = utc_now()
            return AssertionResult(
                assertion_id="state_focused",
                assertion_method="to_be_focused",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Element not found",
                expected_value="focused",
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        is_focused, confidence = self._detector.is_focused(match.bounds, screenshot)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        completed_at = utc_now()

        if is_focused:
            return AssertionResult(
                assertion_id="state_focused",
                assertion_method="to_be_focused",
                status=AssertionStatus.PASSED,
                started_at=started_at,
                completed_at=completed_at,
                expected_value="focused",
                actual_value="focused",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="state_focused",
                assertion_method="to_be_focused",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message=f"Element does not appear focused (confidence: {confidence:.0%})",
                expected_value="focused",
                actual_value="not focused",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_be_checked(
        self,
        screenshot: NDArray[np.uint8],
    ) -> AssertionResult:
        """Assert checkbox/toggle appears checked.

        Args:
            screenshot: Screenshot to analyze.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        start_time = time.monotonic()

        match = await self._locator.find(screenshot)
        if match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            completed_at = utc_now()
            return AssertionResult(
                assertion_id="state_checked",
                assertion_method="to_be_checked",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Element not found",
                expected_value="checked",
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        is_checked, confidence = self._detector.is_checked(match.bounds, screenshot)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        completed_at = utc_now()

        if is_checked:
            return AssertionResult(
                assertion_id="state_checked",
                assertion_method="to_be_checked",
                status=AssertionStatus.PASSED,
                started_at=started_at,
                completed_at=completed_at,
                expected_value="checked",
                actual_value="checked",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="state_checked",
                assertion_method="to_be_checked",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message=f"Element appears unchecked (confidence: {confidence:.0%})",
                expected_value="checked",
                actual_value="unchecked",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_be_unchecked(
        self,
        screenshot: NDArray[np.uint8],
    ) -> AssertionResult:
        """Assert checkbox/toggle appears unchecked.

        Args:
            screenshot: Screenshot to analyze.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        start_time = time.monotonic()

        match = await self._locator.find(screenshot)
        if match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            completed_at = utc_now()
            return AssertionResult(
                assertion_id="state_unchecked",
                assertion_method="to_be_unchecked",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Element not found",
                expected_value="unchecked",
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        is_checked, confidence = self._detector.is_checked(match.bounds, screenshot)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        completed_at = utc_now()

        if not is_checked:
            return AssertionResult(
                assertion_id="state_unchecked",
                assertion_method="to_be_unchecked",
                status=AssertionStatus.PASSED,
                started_at=started_at,
                completed_at=completed_at,
                expected_value="unchecked",
                actual_value="unchecked",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="state_unchecked",
                assertion_method="to_be_unchecked",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message=f"Element appears checked (confidence: {confidence:.0%})",
                expected_value="unchecked",
                actual_value="checked",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

    async def to_have_state(
        self,
        expected_state: str | ElementState,
        screenshot: NDArray[np.uint8],
    ) -> AssertionResult:
        """Assert element has specific visual state.

        Args:
            expected_state: Expected state.
            screenshot: Screenshot to analyze.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        start_time = time.monotonic()

        if isinstance(expected_state, str):
            expected_state = ElementState(expected_state)

        match = await self._locator.find(screenshot)
        if match is None:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            completed_at = utc_now()
            return AssertionResult(
                assertion_id="state_has_state",
                assertion_method="to_have_state",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message="Element not found",
                expected_value=expected_state.value,
                actual_value="not found",
                matches_found=0,
                duration_ms=elapsed_ms,
            )

        result = self._detector.detect_state(match.bounds, screenshot)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        completed_at = utc_now()

        if result.detected_state == expected_state:
            return AssertionResult(
                assertion_id="state_has_state",
                assertion_method="to_have_state",
                status=AssertionStatus.PASSED,
                started_at=started_at,
                completed_at=completed_at,
                expected_value=expected_state.value,
                actual_value=expected_state.value,
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            actual = result.detected_state.value if result.detected_state else "unknown"
            return AssertionResult(
                assertion_id="state_has_state",
                assertion_method="to_have_state",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message=f"Expected state '{expected_state.value}', got '{actual}'",
                expected_value=expected_state.value,
                actual_value=actual,
                matches_found=0,
                duration_ms=elapsed_ms,
            )


__all__ = [
    "ElementState",
    "StateAssertion",
    "StateDetectionResult",
    "StateDetector",
]
