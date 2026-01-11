"""Vision verification expect API.

Provides the main fluent assertion API for DOM-independent visual
assertions using machine vision.

Usage:
    from qontinui.vision.verification import expect

    # Assert element is visible
    await expect(target="button.png").to_be_visible()

    # Assert text content
    await expect(text="Welcome").to_be_visible()
    await expect(region="header").to_have_text("Dashboard")

    # Assert element state
    await expect(target="submit_button").to_be_enabled()
"""

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.assertions import (
    AssertionResult,
    AssertionStatus,
    BoundingBox,
)
from qontinui_schemas.testing.environment import GUIEnvironment

from qontinui.vision.verification.config import VisionConfig, get_default_config
from qontinui.vision.verification.detection.ocr import OCREngine, get_ocr_engine
from qontinui.vision.verification.detection.template import TemplateEngine, get_template_engine
from qontinui.vision.verification.errors import (
    AssertionError,
)
from qontinui.vision.verification.errors import TimeoutError as VisionTimeoutError
from qontinui.vision.verification.locators.base import BaseLocator, LocatorMatch
from qontinui.vision.verification.locators.environment import EnvironmentLocator
from qontinui.vision.verification.locators.image import ImageLocator
from qontinui.vision.verification.locators.region import RegionLocator
from qontinui.vision.verification.locators.text import TextLocator
from qontinui.vision.verification.results import ResultBuilder
from qontinui.vision.verification.screenshot import ScreenshotManager, get_screenshot_manager

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class VisionExpect:
    """Fluent assertion builder for vision verification.

    Provides a Playwright-like expect() API for visual assertions.
    Supports chaining, negation, and soft assertions.

    Usage:
        expect = VisionExpect(locator)
        await expect.to_be_visible()
        await expect.to_have_text("Hello")
        await expect.not_().to_be_disabled()
    """

    def __init__(
        self,
        locator: BaseLocator,
        config: VisionConfig | None = None,
        environment: GUIEnvironment | None = None,
        screenshot_manager: ScreenshotManager | None = None,
        assertion_id: str | None = None,
    ) -> None:
        """Initialize expect builder.

        Args:
            locator: Target element locator.
            config: Vision configuration.
            environment: GUI environment for environment-aware assertions.
            screenshot_manager: Screenshot manager.
            assertion_id: Optional assertion ID.
        """
        self._locator = locator
        self._config = config or get_default_config()
        self._environment = environment
        self._screenshot_manager = screenshot_manager or get_screenshot_manager()
        self._assertion_id = assertion_id or str(uuid.uuid4())[:8]

        # Options
        self._timeout: int | None = None
        self._soft: bool = False
        self._negate: bool = False
        self._message: str | None = None

        # Internal state
        self._screenshot: NDArray[np.uint8] | None = None

        # Detection engines (lazy loaded)
        self._ocr_engine: OCREngine | None = None
        self._template_engine: TemplateEngine | None = None

    def _get_ocr_engine(self) -> OCREngine:
        """Get or create OCR engine.

        Returns:
            OCREngine instance.
        """
        if self._ocr_engine is None:
            self._ocr_engine = get_ocr_engine(
                config=self._config,
                environment=self._environment,
            )
        return self._ocr_engine

    def _get_template_engine(self) -> TemplateEngine:
        """Get or create template engine.

        Returns:
            TemplateEngine instance.
        """
        if self._template_engine is None:
            self._template_engine = get_template_engine(
                config=self._config,
                environment=self._environment,
            )
        return self._template_engine

    def with_timeout(self, timeout_ms: int) -> Self:
        """Set custom timeout.

        Args:
            timeout_ms: Timeout in milliseconds.

        Returns:
            Self for chaining.
        """
        self._timeout = timeout_ms
        return self

    def soft(self) -> Self:
        """Make this a soft assertion (continue on failure).

        Returns:
            Self for chaining.
        """
        self._soft = True
        return self

    def not_(self) -> Self:
        """Negate the assertion.

        Returns:
            Self for chaining.
        """
        self._negate = not self._negate
        return self

    def with_message(self, message: str) -> Self:
        """Set custom failure message.

        Args:
            message: Custom message.

        Returns:
            Self for chaining.
        """
        self._message = message
        return self

    def _get_timeout(self) -> int:
        """Get effective timeout.

        Returns:
            Timeout in milliseconds.
        """
        if self._timeout is not None:
            return self._timeout
        return self._config.wait.default_timeout

    def _get_polling_interval(self) -> int:
        """Get polling interval.

        Returns:
            Polling interval in milliseconds.
        """
        return self._config.wait.polling_interval

    async def _get_screenshot(self, force: bool = False) -> NDArray[np.uint8]:
        """Get current screenshot.

        Args:
            force: Force new capture.

        Returns:
            Screenshot array.
        """
        if self._screenshot is None or force:
            self._screenshot = await self._screenshot_manager.capture(force=force)
        return self._screenshot

    def _handle_result(
        self,
        result: AssertionResult,
        assertion_method: str,
    ) -> AssertionResult:
        """Handle assertion result.

        Args:
            result: Assertion result.
            assertion_method: Method name.

        Returns:
            Result.

        Raises:
            AssertionError: If assertion failed and not soft.
        """
        if result.status == AssertionStatus.FAILED and not self._soft:
            raise AssertionError(
                assertion_method=assertion_method,
                expected=result.expected_value,
                actual=result.actual_value,
                message=result.error_message,
                suggestion=result.suggestion,
                screenshot_path=result.screenshot_path,
                annotated_screenshot_path=result.annotated_screenshot_path,
            )

        return result

    async def _poll_until(
        self,
        check_fn,
        assertion_method: str,
    ) -> AssertionResult:
        """Poll until condition is met or timeout.

        Args:
            check_fn: Async function returning (passed, result_builder).
            assertion_method: Method name for result.

        Returns:
            Assertion result.
        """
        timeout_ms = self._get_timeout()
        polling_ms = self._get_polling_interval()
        start_time = time.time()
        last_result_builder: ResultBuilder | None = None

        while True:
            # Force new screenshot for each poll
            self._screenshot = None
            screenshot = await self._get_screenshot(force=True)

            passed, result_builder = await check_fn(screenshot)

            # Apply negation
            if self._negate:
                passed = not passed

            if passed:
                return self._handle_result(
                    result_builder.set_passed().build(),
                    assertion_method,
                )

            last_result_builder = result_builder

            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms >= timeout_ms:
                # Timeout - build failure result
                if last_result_builder is not None:
                    result = last_result_builder.set_failed().build()

                    # Save failure screenshots
                    if self._config.screenshot.capture_on_failure:
                        screenshot_path, annotated_path = self._screenshot_manager.save_failure(
                            screenshot,
                            assertion_id=self._assertion_id,
                        )
                        result.screenshot_path = str(screenshot_path)
                        if annotated_path:
                            result.annotated_screenshot_path = str(annotated_path)

                    return self._handle_result(result, assertion_method)

                raise VisionTimeoutError(
                    operation=assertion_method,
                    timeout_ms=timeout_ms,
                )

            # Wait before next poll
            await asyncio.sleep(polling_ms / 1000)

    # =========================================================================
    # Visibility Assertions
    # =========================================================================

    async def to_be_visible(self) -> AssertionResult:
        """Assert element is visible on screen.

        Returns:
            Assertion result.

        Raises:
            AssertionError: If assertion fails and not soft.
        """

        async def check(screenshot: NDArray[np.uint8]):
            builder = ResultBuilder(self._assertion_id, "to_be_visible")
            builder.set_expected("element visible")

            matches = await self._locator.find_all(screenshot)

            if matches:
                builder.set_matches(
                    [m.to_schema(self._locator.locator_type, i) for i, m in enumerate(matches)]
                )
                builder.set_actual(f"found {len(matches)} match(es)")
                return True, builder
            else:
                builder.set_actual("not found")
                builder.set_suggestion(
                    "Verify the element is on screen and the locator is correct."
                )
                return False, builder

        return await self._poll_until(check, "to_be_visible")

    async def to_be_hidden(self) -> AssertionResult:
        """Assert element is not visible on screen.

        Returns:
            Assertion result.
        """
        # Use negation internally
        original_negate = self._negate
        self._negate = not self._negate

        try:
            return await self.to_be_visible()
        finally:
            self._negate = original_negate

    # =========================================================================
    # Text Assertions
    # =========================================================================

    async def to_have_text(
        self,
        expected_text: str,
        exact: bool = True,
        case_sensitive: bool = True,
        regex: bool = False,
    ) -> AssertionResult:
        """Assert element/region contains specific text.

        Uses OCREngine with typography hints for improved accuracy.

        Args:
            expected_text: Expected text content.
            exact: Require exact match vs contains.
            case_sensitive: Case-sensitive matching.
            regex: Treat expected_text as regex pattern.

        Returns:
            Assertion result.
        """

        async def check(screenshot: NDArray[np.uint8]):
            builder = ResultBuilder(self._assertion_id, "to_have_text")
            builder.set_expected(expected_text)

            # Find locator match first
            match = await self._locator.find(screenshot)
            if match is None:
                builder.set_actual("element not found")
                builder.set_suggestion("Element must be visible to check text.")
                return False, builder

            # Use OCREngine for text detection with typography hints
            ocr = self._get_ocr_engine()

            # Find specific text in the region
            text_matches = await ocr.find_text(
                target_text=expected_text,
                image=screenshot,
                region=match.bounds,
                exact=exact,
                case_sensitive=case_sensitive,
                regex=regex,
            )

            if text_matches:
                actual_text = text_matches[0].text
                builder.set_actual(actual_text)
                builder.add_detail("confidence", text_matches[0].confidence)
                builder.add_detail("font_size", text_matches[0].font_size_estimate)
                return True, builder
            else:
                # Extract all text to show what was found
                all_text = await ocr.extract_text_from_region(
                    screenshot, match.bounds, join_lines=True
                )

                if all_text:
                    builder.set_actual(f"found: {all_text[:100]}")
                else:
                    builder.set_actual("no text found")

                builder.set_suggestion(
                    f"Expected '{expected_text}' but found '{all_text[:50]}'"
                    if all_text
                    else "No text detected in region."
                )
                return False, builder

        return await self._poll_until(check, "to_have_text")

    async def to_contain_text(
        self,
        expected_text: str,
        case_sensitive: bool = True,
    ) -> AssertionResult:
        """Assert element/region contains text (partial match).

        Args:
            expected_text: Text to search for.
            case_sensitive: Case-sensitive matching.

        Returns:
            Assertion result.
        """
        return await self.to_have_text(expected_text, exact=False, case_sensitive=case_sensitive)

    async def to_match_text(
        self,
        pattern: str,
        case_sensitive: bool = True,
    ) -> AssertionResult:
        """Assert element/region text matches regex pattern.

        Args:
            pattern: Regex pattern to match.
            case_sensitive: Case-sensitive matching.

        Returns:
            Assertion result.
        """
        return await self.to_have_text(
            pattern, exact=False, case_sensitive=case_sensitive, regex=True
        )

    # =========================================================================
    # Count Assertions
    # =========================================================================

    async def to_have_count(self, count: int) -> AssertionResult:
        """Assert number of matching elements.

        Args:
            count: Expected count.

        Returns:
            Assertion result.
        """

        async def check(screenshot: NDArray[np.uint8]):
            builder = ResultBuilder(self._assertion_id, "to_have_count")
            builder.set_expected(count)

            matches = await self._locator.find_all(screenshot)
            actual_count = len(matches)
            builder.set_actual(actual_count)

            if matches:
                builder.set_matches(
                    [m.to_schema(self._locator.locator_type, i) for i, m in enumerate(matches)]
                )

            if actual_count == count:
                return True, builder
            else:
                builder.set_suggestion(f"Expected {count} elements but found {actual_count}.")
                return False, builder

        return await self._poll_until(check, "to_have_count")

    async def to_have_count_between(
        self,
        min_count: int,
        max_count: int,
    ) -> AssertionResult:
        """Assert count is within range.

        Args:
            min_count: Minimum expected count.
            max_count: Maximum expected count.

        Returns:
            Assertion result.
        """

        async def check(screenshot: NDArray[np.uint8]):
            builder = ResultBuilder(self._assertion_id, "to_have_count_between")
            builder.set_expected(f"{min_count}-{max_count}")

            matches = await self._locator.find_all(screenshot)
            actual_count = len(matches)
            builder.set_actual(actual_count)

            if matches:
                builder.set_matches(
                    [m.to_schema(self._locator.locator_type, i) for i, m in enumerate(matches)]
                )

            if min_count <= actual_count <= max_count:
                return True, builder
            else:
                return False, builder

        return await self._poll_until(check, "to_have_count_between")

    async def to_have_count_at_least(self, min_count: int) -> AssertionResult:
        """Assert at least N matching elements.

        Args:
            min_count: Minimum expected count.

        Returns:
            Assertion result.
        """

        async def check(screenshot: NDArray[np.uint8]):
            builder = ResultBuilder(self._assertion_id, "to_have_count_at_least")
            builder.set_expected(f">={min_count}")

            matches = await self._locator.find_all(screenshot)
            actual_count = len(matches)
            builder.set_actual(actual_count)

            if matches:
                builder.set_matches(
                    [m.to_schema(self._locator.locator_type, i) for i, m in enumerate(matches)]
                )

            if actual_count >= min_count:
                return True, builder
            else:
                builder.set_suggestion(
                    f"Expected at least {min_count} elements but found {actual_count}."
                )
                return False, builder

        return await self._poll_until(check, "to_have_count_at_least")

    async def to_have_count_at_most(self, max_count: int) -> AssertionResult:
        """Assert at most N matching elements.

        Args:
            max_count: Maximum expected count.

        Returns:
            Assertion result.
        """

        async def check(screenshot: NDArray[np.uint8]):
            builder = ResultBuilder(self._assertion_id, "to_have_count_at_most")
            builder.set_expected(f"<={max_count}")

            matches = await self._locator.find_all(screenshot)
            actual_count = len(matches)
            builder.set_actual(actual_count)

            if matches:
                builder.set_matches(
                    [m.to_schema(self._locator.locator_type, i) for i, m in enumerate(matches)]
                )

            if actual_count <= max_count:
                return True, builder
            else:
                builder.set_suggestion(
                    f"Expected at most {max_count} elements but found {actual_count}."
                )
                return False, builder

        return await self._poll_until(check, "to_have_count_at_most")

    async def to_be_empty(self) -> AssertionResult:
        """Assert no matching elements (count == 0).

        Returns:
            Assertion result.
        """
        return await self.to_have_count(0)

    async def to_not_be_empty(self) -> AssertionResult:
        """Assert at least one matching element.

        Returns:
            Assertion result.
        """
        return await self.to_have_count_at_least(1)

    # =========================================================================
    # State Assertions
    # =========================================================================

    async def to_be_enabled(self) -> AssertionResult:
        """Assert element appears enabled.

        Uses learned visual states if environment is available.

        Returns:
            Assertion result.
        """
        return await self._check_visual_state("enabled")

    async def to_be_disabled(self) -> AssertionResult:
        """Assert element appears disabled.

        Returns:
            Assertion result.
        """
        return await self._check_visual_state("disabled")

    async def to_be_focused(self) -> AssertionResult:
        """Assert element appears focused.

        Returns:
            Assertion result.
        """
        return await self._check_visual_state("focused")

    async def to_be_checked(self) -> AssertionResult:
        """Assert checkbox/toggle appears checked.

        Returns:
            Assertion result.
        """
        return await self._check_visual_state("checked")

    async def _check_visual_state(self, state_name: str) -> AssertionResult:
        """Check visual state using environment or heuristics.

        Args:
            state_name: State to check (enabled, disabled, focused, checked).

        Returns:
            Assertion result.
        """

        async def check(screenshot: NDArray[np.uint8]):
            builder = ResultBuilder(self._assertion_id, f"to_be_{state_name}")
            builder.set_expected(state_name)

            # Find element
            match = await self._locator.find(screenshot)
            if match is None:
                builder.set_actual("element not found")
                return False, builder

            # Check state using environment if available
            if self._environment is not None and self._config.environment.use_learned_states:
                is_state = self._check_state_with_environment(screenshot, match, state_name)
                builder.set_actual(state_name if is_state else f"not {state_name}")
                return is_state, builder

            # Fall back to heuristics
            is_state = self._check_state_heuristic(screenshot, match, state_name)
            builder.set_actual(state_name if is_state else f"not {state_name}")
            builder.add_detail("method", "heuristic")

            return is_state, builder

        return await self._poll_until(check, f"to_be_{state_name}")

    def _check_state_with_environment(
        self,
        screenshot: NDArray[np.uint8],
        match: LocatorMatch,
        state_name: str,
    ) -> bool:
        """Check state using environment-learned visual signatures.

        Args:
            screenshot: Current screenshot.
            match: Element match.
            state_name: State to check.

        Returns:
            True if element appears to be in the specified state.
        """
        # This would use the visual_states from GUIEnvironment
        # For now, fall back to heuristics
        return self._check_state_heuristic(screenshot, match, state_name)

    def _check_state_heuristic(
        self,
        screenshot: NDArray[np.uint8],
        match: LocatorMatch,
        state_name: str,
    ) -> bool:
        """Check state using visual heuristics.

        Args:
            screenshot: Current screenshot.
            match: Element match.
            state_name: State to check.

        Returns:
            True if element appears to be in the specified state.
        """
        import cv2

        # Extract element region
        b = match.bounds
        region = screenshot[b.y : b.y + b.height, b.x : b.x + b.width]

        if region.size == 0:
            return False

        # Convert to HSV for saturation analysis
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].mean()
        brightness = hsv[:, :, 2].mean()

        if state_name == "disabled":
            # Disabled elements typically have low saturation
            return bool(saturation < 50 and brightness < 200)

        elif state_name == "enabled":
            # Enabled elements typically have normal saturation
            return bool(saturation >= 50 or brightness >= 200)

        elif state_name == "focused":
            # Check for focus ring (bright border)
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_ratio = edges.sum() / (region.shape[0] * region.shape[1] * 255)
            return bool(edge_ratio > 0.1)

        elif state_name == "checked":
            # Check for interior fill
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            fill_ratio = thresh.sum() / (region.shape[0] * region.shape[1] * 255)
            return bool(fill_ratio > 0.3)

        return False

    # =========================================================================
    # Spatial Assertions
    # =========================================================================

    async def to_be_above(self, reference: BaseLocator) -> AssertionResult:
        """Assert element is above reference element.

        Args:
            reference: Reference element locator.

        Returns:
            Assertion result.
        """
        return await self._check_spatial_relation("above", reference)

    async def to_be_below(self, reference: BaseLocator) -> AssertionResult:
        """Assert element is below reference element.

        Args:
            reference: Reference element locator.

        Returns:
            Assertion result.
        """
        return await self._check_spatial_relation("below", reference)

    async def to_be_left_of(self, reference: BaseLocator) -> AssertionResult:
        """Assert element is left of reference element.

        Args:
            reference: Reference element locator.

        Returns:
            Assertion result.
        """
        return await self._check_spatial_relation("left_of", reference)

    async def to_be_right_of(self, reference: BaseLocator) -> AssertionResult:
        """Assert element is right of reference element.

        Args:
            reference: Reference element locator.

        Returns:
            Assertion result.
        """
        return await self._check_spatial_relation("right_of", reference)

    async def _check_spatial_relation(
        self,
        relation: str,
        reference: BaseLocator,
    ) -> AssertionResult:
        """Check spatial relationship between elements.

        Args:
            relation: Relationship type.
            reference: Reference element locator.

        Returns:
            Assertion result.
        """

        async def check(screenshot: NDArray[np.uint8]):
            builder = ResultBuilder(self._assertion_id, f"to_be_{relation}")
            builder.set_expected(relation)

            # Find both elements
            target_match = await self._locator.find(screenshot)
            ref_match = await reference.find(screenshot)

            if target_match is None:
                builder.set_actual("target not found")
                return False, builder

            if ref_match is None:
                builder.set_actual("reference not found")
                return False, builder

            # Check relation
            t = target_match.center
            r = ref_match.center

            result = False
            if relation == "above":
                result = t[1] < r[1]
            elif relation == "below":
                result = t[1] > r[1]
            elif relation == "left_of":
                result = t[0] < r[0]
            elif relation == "right_of":
                result = t[0] > r[0]

            builder.set_actual(f"target at {t}, reference at {r}")
            return result, builder

        return await self._poll_until(check, f"to_be_{relation}")


# =============================================================================
# Factory Functions
# =============================================================================


def expect(
    target: str | None = None,
    text: str | None = None,
    region: BoundingBox | dict[str, int] | None = None,
    state: str | None = None,
    locator: BaseLocator | None = None,
    config: VisionConfig | None = None,
    environment: GUIEnvironment | None = None,
    **options: Any,
) -> VisionExpect:
    """Create a vision expectation.

    This is the main entry point for vision assertions.

    Args:
        target: Image path for template matching.
        text: Text to find via OCR.
        region: Region bounds (dict with x, y, width, height).
        state: Qontinui state name.
        locator: Pre-built locator.
        config: Vision configuration.
        environment: GUI environment.
        **options: Additional locator options.

    Returns:
        VisionExpect instance.

    Raises:
        ValueError: If no target type specified.

    Examples:
        # Template matching
        await expect(target="button.png").to_be_visible()

        # Text matching
        await expect(text="Submit").to_be_visible()

        # Region
        await expect(region={"x": 100, "y": 200, "width": 50, "height": 30}).to_have_text("OK")

        # Pre-built locator
        await expect(locator=my_locator).to_be_visible()
    """
    config = config or get_default_config()

    # Create locator based on target type
    if locator is not None:
        loc = locator
    elif target is not None:
        loc = ImageLocator(target, config=config, **options)
    elif text is not None:
        loc = TextLocator(text, config=config, **options)
    elif region is not None:
        if isinstance(region, dict):
            region = BoundingBox(**region)
        loc = RegionLocator.from_bounds(region, config=config, **options)
    elif state is not None:
        # State locator would use qontinui state machine
        raise NotImplementedError("State locator not yet implemented")
    else:
        raise ValueError("Must specify one of: target, text, region, state, or locator")

    return VisionExpect(
        locator=loc,
        config=config,
        environment=environment,
    )


# Alias for Playwright-like API
def locator(
    image: str | None = None,
    text: str | None = None,
    region: BoundingBox | dict[str, int] | None = None,
    semantic: str | None = None,
    config: VisionConfig | None = None,
    environment: GUIEnvironment | None = None,
    **options: Any,
) -> BaseLocator:
    """Create a vision locator.

    Args:
        image: Image path for template matching.
        text: Text to find via OCR.
        region: Region bounds.
        semantic: Semantic locator description.
        config: Vision configuration.
        environment: GUI environment.
        **options: Additional locator options.

    Returns:
        BaseLocator instance.

    Raises:
        ValueError: If no locator type specified.
    """
    config = config or get_default_config()

    if image is not None:
        return ImageLocator(image, config=config, **options)
    elif text is not None:
        return TextLocator(text, config=config, **options)
    elif region is not None:
        if isinstance(region, dict):
            region = BoundingBox(**region)
        return RegionLocator.from_bounds(region, config=config, **options)
    elif semantic is not None and environment is not None:
        return EnvironmentLocator.element_pattern(semantic, environment=environment, config=config)
    else:
        raise ValueError("Must specify one of: image, text, region, or semantic")


__all__ = [
    "VisionExpect",
    "expect",
    "locator",
]
