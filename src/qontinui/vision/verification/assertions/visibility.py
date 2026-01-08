"""Visibility assertions for vision verification.

Provides to_be_visible and to_be_hidden assertions with
environment-aware detection and configurable wait behavior.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.assertions import (
    AssertionResult,
    AssertionStatus,
    AssertionType,
    BoundingBox,
    VisionLocatorMatch,
)

if TYPE_CHECKING:
    from qontinui.vision.verification.config import VisionConfig
    from qontinui.vision.verification.locators.base import BaseLocator

logger = logging.getLogger(__name__)


class VisibilityAssertion:
    """Assertion for element visibility.

    Checks whether an element is visible or hidden on screen
    using the locator's detection capabilities.

    Usage:
        locator = ImageLocator("button.png")
        assertion = VisibilityAssertion(locator, config)

        # Check visible
        result = await assertion.to_be_visible(screenshot)

        # Check hidden
        result = await assertion.to_be_hidden(screenshot)

        # With wait
        result = await assertion.to_be_visible(screenshot, timeout_ms=5000)
    """

    def __init__(
        self,
        locator: "BaseLocator",
        config: "VisionConfig | None" = None,
    ) -> None:
        """Initialize visibility assertion.

        Args:
            locator: Locator for finding the element.
            config: Vision configuration.
        """
        self._locator = locator
        self._config = config

    def _get_timeout(self, timeout_ms: int | None) -> int:
        """Get timeout value.

        Args:
            timeout_ms: Override timeout.

        Returns:
            Timeout in milliseconds.
        """
        if timeout_ms is not None:
            return timeout_ms
        if self._config is not None:
            return self._config.wait.default_timeout_ms
        return 5000

    def _get_poll_interval(self) -> int:
        """Get poll interval.

        Returns:
            Poll interval in milliseconds.
        """
        if self._config is not None:
            return self._config.wait.poll_interval_ms
        return 100

    async def to_be_visible(
        self,
        screenshot: NDArray[np.uint8],
        screenshot_callback: Any | None = None,
        timeout_ms: int | None = None,
        region: BoundingBox | None = None,
    ) -> AssertionResult:
        """Assert element is visible.

        Args:
            screenshot: Current screenshot.
            screenshot_callback: Async function to get fresh screenshots for retries.
            timeout_ms: Maximum wait time.
            region: Optional region to limit search.

        Returns:
            Assertion result.
        """
        timeout = self._get_timeout(timeout_ms)
        poll_interval = self._get_poll_interval()

        start_time = time.monotonic()
        deadline = start_time + (timeout / 1000.0)
        attempts = 0
        last_error: str | None = None
        current_screenshot = screenshot

        while True:
            attempts += 1
            try:
                # Find matches
                if region is not None:
                    self._locator.inside(region)

                matches = await self._locator.find_all(current_screenshot)

                if matches:
                    # Element is visible
                    elapsed_ms = int((time.monotonic() - start_time) * 1000)

                    best_match = max(matches, key=lambda m: m.confidence)

                    return AssertionResult(
                        assertion_id="visibility_visible",
                        locator_value=self._locator._value,
                        assertion_type=AssertionType.VISIBILITY,
                        assertion_method="to_be_visible",
                        status=AssertionStatus.PASSED,
                        message=f"Element is visible ({len(matches)} match(es) found)",
                        expected_value=True,
                        actual_value=True,
                        matches_found=len(matches),
                        best_match=VisionLocatorMatch(
                            bounds=best_match.bounds,
                            confidence=best_match.confidence,
                            text=best_match.text,
                        ),
                        duration_ms=elapsed_ms,
                        attempts=attempts,
                    )

                last_error = "No matches found"

            except Exception as e:
                last_error = str(e)
                logger.debug(f"Visibility check failed: {e}")

            # Check timeout
            now = time.monotonic()
            if now >= deadline:
                elapsed_ms = int((now - start_time) * 1000)

                return AssertionResult(
                    assertion_id="visibility_visible",
                    locator_value=self._locator._value,
                    assertion_type=AssertionType.VISIBILITY,
                    assertion_method="to_be_visible",
                    status=AssertionStatus.FAILED,
                    message=f"Element not visible after {elapsed_ms}ms: {last_error}",
                    expected_value=True,
                    actual_value=False,
                    matches_found=0,
                    duration_ms=elapsed_ms,
                    attempts=attempts,
                )

            # Wait and get fresh screenshot
            await asyncio.sleep(poll_interval / 1000.0)

            if screenshot_callback is not None:
                try:
                    current_screenshot = await screenshot_callback()
                except Exception as e:
                    logger.warning(f"Failed to get fresh screenshot: {e}")

    async def to_be_hidden(
        self,
        screenshot: NDArray[np.uint8],
        screenshot_callback: Any | None = None,
        timeout_ms: int | None = None,
        region: BoundingBox | None = None,
    ) -> AssertionResult:
        """Assert element is hidden (not visible).

        Args:
            screenshot: Current screenshot.
            screenshot_callback: Async function to get fresh screenshots for retries.
            timeout_ms: Maximum wait time.
            region: Optional region to limit search.

        Returns:
            Assertion result.
        """
        timeout = self._get_timeout(timeout_ms)
        poll_interval = self._get_poll_interval()

        start_time = time.monotonic()
        deadline = start_time + (timeout / 1000.0)
        attempts = 0
        current_screenshot = screenshot

        while True:
            attempts += 1
            try:
                # Find matches
                if region is not None:
                    self._locator.inside(region)

                matches = await self._locator.find_all(current_screenshot)

                if not matches:
                    # Element is hidden
                    elapsed_ms = int((time.monotonic() - start_time) * 1000)

                    return AssertionResult(
                        assertion_id="visibility_hidden",
                        locator_value=self._locator._value,
                        assertion_type=AssertionType.VISIBILITY,
                        assertion_method="to_be_hidden",
                        status=AssertionStatus.PASSED,
                        message="Element is hidden (not found)",
                        expected_value=False,
                        actual_value=False,
                        matches_found=0,
                        duration_ms=elapsed_ms,
                        attempts=attempts,
                    )

            except Exception:
                # Exception during search means element not found = hidden
                elapsed_ms = int((time.monotonic() - start_time) * 1000)

                return AssertionResult(
                    assertion_id="visibility_hidden",
                    locator_value=self._locator._value,
                    assertion_type=AssertionType.VISIBILITY,
                    assertion_method="to_be_hidden",
                    status=AssertionStatus.PASSED,
                    message="Element is hidden (search failed)",
                    expected_value=False,
                    actual_value=False,
                    matches_found=0,
                    duration_ms=elapsed_ms,
                    attempts=attempts,
                )

            # Check timeout
            now = time.monotonic()
            if now >= deadline:
                elapsed_ms = int((now - start_time) * 1000)

                best_match = max(matches, key=lambda m: m.confidence)

                return AssertionResult(
                    assertion_id="visibility_hidden",
                    locator_value=self._locator._value,
                    assertion_type=AssertionType.VISIBILITY,
                    assertion_method="to_be_hidden",
                    status=AssertionStatus.FAILED,
                    message=f"Element still visible after {elapsed_ms}ms ({len(matches)} match(es))",
                    expected_value=False,
                    actual_value=True,
                    matches_found=len(matches),
                    best_match=VisionLocatorMatch(
                        bounds=best_match.bounds,
                        confidence=best_match.confidence,
                        text=best_match.text,
                    ),
                    duration_ms=elapsed_ms,
                    attempts=attempts,
                )

            # Wait and get fresh screenshot
            await asyncio.sleep(poll_interval / 1000.0)

            if screenshot_callback is not None:
                try:
                    current_screenshot = await screenshot_callback()
                except Exception as e:
                    logger.warning(f"Failed to get fresh screenshot: {e}")


__all__ = ["VisibilityAssertion"]
