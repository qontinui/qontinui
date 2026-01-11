"""Count assertions for vision verification.

Provides to_have_count assertion for verifying the number
of matching elements on screen.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.common import utc_now
from qontinui_schemas.testing.assertions import (
    AssertionResult,
    AssertionStatus,
    AssertionType,
    BoundingBox,
    LocatorType,
    VisionLocatorMatch,
)

if TYPE_CHECKING:
    from qontinui.vision.verification.config import VisionConfig
    from qontinui.vision.verification.locators.base import BaseLocator

logger = logging.getLogger(__name__)


class CountAssertion:
    """Assertion for element count.

    Checks the number of elements matching a locator
    with support for exact, minimum, maximum, and range counts.

    Usage:
        locator = ImageLocator("list_item.png")
        assertion = CountAssertion(locator, config)

        # Exact count
        result = await assertion.to_have_count(5, screenshot)

        # At least N
        result = await assertion.to_have_count_at_least(3, screenshot)

        # At most N
        result = await assertion.to_have_count_at_most(10, screenshot)

        # Range
        result = await assertion.to_have_count_between(3, 10, screenshot)
    """

    def __init__(
        self,
        locator: "BaseLocator",
        config: "VisionConfig | None" = None,
    ) -> None:
        """Initialize count assertion.

        Args:
            locator: Locator for finding elements.
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
            return self._config.wait.default_timeout
        return 5000

    def _get_poll_interval(self) -> int:
        """Get poll interval.

        Returns:
            Poll interval in milliseconds.
        """
        if self._config is not None:
            return self._config.wait.polling_interval
        return 100

    async def to_have_count(
        self,
        expected_count: int,
        screenshot: NDArray[np.uint8],
        screenshot_callback: Any | None = None,
        timeout_ms: int | None = None,
        region: BoundingBox | None = None,
    ) -> AssertionResult:
        """Assert exact count of matching elements.

        Args:
            expected_count: Expected number of matches.
            screenshot: Current screenshot.
            screenshot_callback: Async function to get fresh screenshots.
            timeout_ms: Maximum wait time.
            region: Optional region to limit search.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        timeout = self._get_timeout(timeout_ms)
        poll_interval = self._get_poll_interval()

        start_time = time.monotonic()
        deadline = start_time + (timeout / 1000.0)
        attempts = 0
        last_count = 0
        current_screenshot = screenshot

        while True:
            attempts += 1
            try:
                # Find matches
                if region is not None:
                    self._locator.with_region(region)

                matches = await self._locator.find_all(current_screenshot)
                last_count = len(matches)

                if last_count == expected_count:
                    elapsed_ms = int((time.monotonic() - start_time) * 1000)

                    best_match = None
                    if matches:
                        best = max(matches, key=lambda m: m.confidence)
                        center_x = best.bounds.x + best.bounds.width // 2
                        center_y = best.bounds.y + best.bounds.height // 2
                        best_match = VisionLocatorMatch(
                            bounds=best.bounds,
                            confidence=best.confidence,
                            center=(center_x, center_y),
                            text=best.text,
                            locator_type=LocatorType.IMAGE,
                        )

                    completed_at = utc_now()
                    return AssertionResult(
                        assertion_id="count_exact",
                        assertion_method="to_have_count",
                        status=AssertionStatus.PASSED,
                        started_at=started_at,
                        completed_at=completed_at,
                        expected_value=expected_count,
                        actual_value=last_count,
                        matches_found=last_count,
                        best_match=best_match,
                        duration_ms=elapsed_ms,
                        retry_count=attempts - 1,
                    )

            except Exception as e:
                logger.debug(f"Count assertion failed: {e}")

            # Check timeout
            now = time.monotonic()
            if now >= deadline:
                elapsed_ms = int((now - start_time) * 1000)

                completed_at = utc_now()
                return AssertionResult(
                    assertion_id="count_exact",
                    assertion_method="to_have_count",
                    status=AssertionStatus.FAILED,
                    started_at=started_at,
                    completed_at=completed_at,
                    error_message=f"Count mismatch: expected {expected_count}, found {last_count}",
                    expected_value=expected_count,
                    actual_value=last_count,
                    matches_found=last_count,
                    duration_ms=elapsed_ms,
                    retry_count=attempts - 1,
                )

            # Wait and get fresh screenshot
            await asyncio.sleep(poll_interval / 1000.0)

            if screenshot_callback is not None:
                try:
                    current_screenshot = await screenshot_callback()
                except Exception as e:
                    logger.warning(f"Failed to get fresh screenshot: {e}")

    async def to_have_count_at_least(
        self,
        min_count: int,
        screenshot: NDArray[np.uint8],
        screenshot_callback: Any | None = None,
        timeout_ms: int | None = None,
        region: BoundingBox | None = None,
    ) -> AssertionResult:
        """Assert at least N matching elements.

        Args:
            min_count: Minimum number of matches.
            screenshot: Current screenshot.
            screenshot_callback: Async function to get fresh screenshots.
            timeout_ms: Maximum wait time.
            region: Optional region to limit search.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        timeout = self._get_timeout(timeout_ms)
        poll_interval = self._get_poll_interval()

        start_time = time.monotonic()
        deadline = start_time + (timeout / 1000.0)
        attempts = 0
        last_count = 0
        current_screenshot = screenshot

        while True:
            attempts += 1
            try:
                if region is not None:
                    self._locator.with_region(region)

                matches = await self._locator.find_all(current_screenshot)
                last_count = len(matches)

                if last_count >= min_count:
                    elapsed_ms = int((time.monotonic() - start_time) * 1000)

                    best_match = None
                    if matches:
                        best = max(matches, key=lambda m: m.confidence)
                        center_x = best.bounds.x + best.bounds.width // 2
                        center_y = best.bounds.y + best.bounds.height // 2
                        best_match = VisionLocatorMatch(
                            bounds=best.bounds,
                            confidence=best.confidence,
                            center=(center_x, center_y),
                            text=best.text,
                            locator_type=LocatorType.IMAGE,
                        )

                    completed_at = utc_now()
                    return AssertionResult(
                        assertion_id="count_at_least",
                        assertion_method="to_have_count_at_least",
                        status=AssertionStatus.PASSED,
                        started_at=started_at,
                        completed_at=completed_at,
                        expected_value=f">={min_count}",
                        actual_value=last_count,
                        matches_found=last_count,
                        best_match=best_match,
                        duration_ms=elapsed_ms,
                        retry_count=attempts - 1,
                    )

            except Exception as e:
                logger.debug(f"Count assertion failed: {e}")

            # Check timeout
            now = time.monotonic()
            if now >= deadline:
                elapsed_ms = int((now - start_time) * 1000)

                completed_at = utc_now()
                return AssertionResult(
                    assertion_id="count_at_least",
                    assertion_method="to_have_count_at_least",
                    status=AssertionStatus.FAILED,
                    started_at=started_at,
                    completed_at=completed_at,
                    error_message=f"Found {last_count} element(s), expected at least {min_count}",
                    expected_value=f">={min_count}",
                    actual_value=last_count,
                    matches_found=last_count,
                    duration_ms=elapsed_ms,
                    retry_count=attempts - 1,
                )

            await asyncio.sleep(poll_interval / 1000.0)

            if screenshot_callback is not None:
                try:
                    current_screenshot = await screenshot_callback()
                except Exception as e:
                    logger.warning(f"Failed to get fresh screenshot: {e}")

    async def to_have_count_at_most(
        self,
        max_count: int,
        screenshot: NDArray[np.uint8],
        screenshot_callback: Any | None = None,
        timeout_ms: int | None = None,
        region: BoundingBox | None = None,
    ) -> AssertionResult:
        """Assert at most N matching elements.

        Args:
            max_count: Maximum number of matches.
            screenshot: Current screenshot.
            screenshot_callback: Async function to get fresh screenshots.
            timeout_ms: Maximum wait time.
            region: Optional region to limit search.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        timeout = self._get_timeout(timeout_ms)
        poll_interval = self._get_poll_interval()

        start_time = time.monotonic()
        deadline = start_time + (timeout / 1000.0)
        attempts = 0
        last_count = 0
        current_screenshot = screenshot

        while True:
            attempts += 1
            try:
                if region is not None:
                    self._locator.with_region(region)

                matches = await self._locator.find_all(current_screenshot)
                last_count = len(matches)

                if last_count <= max_count:
                    elapsed_ms = int((time.monotonic() - start_time) * 1000)

                    best_match = None
                    if matches:
                        best = max(matches, key=lambda m: m.confidence)
                        center_x = best.bounds.x + best.bounds.width // 2
                        center_y = best.bounds.y + best.bounds.height // 2
                        best_match = VisionLocatorMatch(
                            bounds=best.bounds,
                            confidence=best.confidence,
                            center=(center_x, center_y),
                            text=best.text,
                            locator_type=LocatorType.IMAGE,
                        )

                    completed_at = utc_now()
                    return AssertionResult(
                        assertion_id="count_at_most",
                        assertion_method="to_have_count_at_most",
                        status=AssertionStatus.PASSED,
                        started_at=started_at,
                        completed_at=completed_at,
                        expected_value=f"<={max_count}",
                        actual_value=last_count,
                        matches_found=last_count,
                        best_match=best_match,
                        duration_ms=elapsed_ms,
                        retry_count=attempts - 1,
                    )

            except Exception as e:
                logger.debug(f"Count assertion failed: {e}")

            # Check timeout
            now = time.monotonic()
            if now >= deadline:
                elapsed_ms = int((now - start_time) * 1000)

                completed_at = utc_now()
                return AssertionResult(
                    assertion_id="count_at_most",
                    assertion_method="to_have_count_at_most",
                    status=AssertionStatus.FAILED,
                    started_at=started_at,
                    completed_at=completed_at,
                    error_message=f"Found {last_count} element(s), expected at most {max_count}",
                    expected_value=f"<={max_count}",
                    actual_value=last_count,
                    matches_found=last_count,
                    duration_ms=elapsed_ms,
                    retry_count=attempts - 1,
                )

            await asyncio.sleep(poll_interval / 1000.0)

            if screenshot_callback is not None:
                try:
                    current_screenshot = await screenshot_callback()
                except Exception as e:
                    logger.warning(f"Failed to get fresh screenshot: {e}")

    async def to_have_count_between(
        self,
        min_count: int,
        max_count: int,
        screenshot: NDArray[np.uint8],
        screenshot_callback: Any | None = None,
        timeout_ms: int | None = None,
        region: BoundingBox | None = None,
    ) -> AssertionResult:
        """Assert count is between min and max (inclusive).

        Args:
            min_count: Minimum number of matches.
            max_count: Maximum number of matches.
            screenshot: Current screenshot.
            screenshot_callback: Async function to get fresh screenshots.
            timeout_ms: Maximum wait time.
            region: Optional region to limit search.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        timeout = self._get_timeout(timeout_ms)
        poll_interval = self._get_poll_interval()

        start_time = time.monotonic()
        deadline = start_time + (timeout / 1000.0)
        attempts = 0
        last_count = 0
        current_screenshot = screenshot

        while True:
            attempts += 1
            try:
                if region is not None:
                    self._locator.with_region(region)

                matches = await self._locator.find_all(current_screenshot)
                last_count = len(matches)

                if min_count <= last_count <= max_count:
                    elapsed_ms = int((time.monotonic() - start_time) * 1000)

                    best_match = None
                    if matches:
                        best = max(matches, key=lambda m: m.confidence)
                        center_x = best.bounds.x + best.bounds.width // 2
                        center_y = best.bounds.y + best.bounds.height // 2
                        best_match = VisionLocatorMatch(
                            bounds=best.bounds,
                            confidence=best.confidence,
                            center=(center_x, center_y),
                            text=best.text,
                            locator_type=LocatorType.IMAGE,
                        )

                    completed_at = utc_now()
                    return AssertionResult(
                        assertion_id="count_between",
                        assertion_method="to_have_count_between",
                        status=AssertionStatus.PASSED,
                        started_at=started_at,
                        completed_at=completed_at,
                        expected_value=f"{min_count}-{max_count}",
                        actual_value=last_count,
                        matches_found=last_count,
                        best_match=best_match,
                        duration_ms=elapsed_ms,
                        retry_count=attempts - 1,
                    )

            except Exception as e:
                logger.debug(f"Count assertion failed: {e}")

            # Check timeout
            now = time.monotonic()
            if now >= deadline:
                elapsed_ms = int((now - start_time) * 1000)

                completed_at = utc_now()
                return AssertionResult(
                    assertion_id="count_between",
                    assertion_method="to_have_count_between",
                    status=AssertionStatus.FAILED,
                    started_at=started_at,
                    completed_at=completed_at,
                    error_message=f"Found {last_count} element(s), expected {min_count}-{max_count}",
                    expected_value=f"{min_count}-{max_count}",
                    actual_value=last_count,
                    matches_found=last_count,
                    duration_ms=elapsed_ms,
                    retry_count=attempts - 1,
                )

            await asyncio.sleep(poll_interval / 1000.0)

            if screenshot_callback is not None:
                try:
                    current_screenshot = await screenshot_callback()
                except Exception as e:
                    logger.warning(f"Failed to get fresh screenshot: {e}")

    async def to_be_empty(
        self,
        screenshot: NDArray[np.uint8],
        screenshot_callback: Any | None = None,
        timeout_ms: int | None = None,
        region: BoundingBox | None = None,
    ) -> AssertionResult:
        """Assert no matching elements (count == 0).

        Convenience method for to_have_count(0).

        Args:
            screenshot: Current screenshot.
            screenshot_callback: Async function to get fresh screenshots.
            timeout_ms: Maximum wait time.
            region: Optional region to limit search.

        Returns:
            Assertion result.
        """
        result = await self.to_have_count(
            expected_count=0,
            screenshot=screenshot,
            screenshot_callback=screenshot_callback,
            timeout_ms=timeout_ms,
            region=region,
        )

        # Update method name
        result.assertion_method = "to_be_empty"
        result.assertion_id = "count_empty"
        if result.status != AssertionStatus.PASSED:
            result.error_message = f"Expected empty, found {result.actual_value} element(s)"

        return result

    async def to_not_be_empty(
        self,
        screenshot: NDArray[np.uint8],
        screenshot_callback: Any | None = None,
        timeout_ms: int | None = None,
        region: BoundingBox | None = None,
    ) -> AssertionResult:
        """Assert at least one matching element (count >= 1).

        Convenience method for to_have_count_at_least(1).

        Args:
            screenshot: Current screenshot.
            screenshot_callback: Async function to get fresh screenshots.
            timeout_ms: Maximum wait time.
            region: Optional region to limit search.

        Returns:
            Assertion result.
        """
        result = await self.to_have_count_at_least(
            min_count=1,
            screenshot=screenshot,
            screenshot_callback=screenshot_callback,
            timeout_ms=timeout_ms,
            region=region,
        )

        # Update method name
        result.assertion_method = "to_not_be_empty"
        result.assertion_id = "count_not_empty"
        if result.status != AssertionStatus.PASSED:
            result.error_message = "No elements found (empty)"

        return result


__all__ = ["CountAssertion"]
