"""Text assertions for vision verification.

Provides to_have_text and to_contain_text assertions using
OCR with environment-aware typography hints.
"""

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.common import utc_now
from qontinui_schemas.testing.assertions import (
    AssertionResult,
    AssertionStatus,
    BoundingBox,
    LocatorType,
    VisionLocatorMatch,
)

from qontinui.vision.verification.detection.ocr import OCREngine, get_ocr_engine

if TYPE_CHECKING:
    from qontinui_schemas.testing.environment import GUIEnvironment

    from qontinui.vision.verification.config import VisionConfig
    from qontinui.vision.verification.locators.base import BaseLocator

logger = logging.getLogger(__name__)


class TextAssertion:
    """Assertion for text content.

    Checks text content within elements or regions using OCR
    with environment-aware typography hints.

    Usage:
        locator = RegionLocator((100, 100, 200, 50))
        assertion = TextAssertion(locator, config)

        # Exact match
        result = await assertion.to_have_text("Submit", screenshot)

        # Contains
        result = await assertion.to_contain_text("Order", screenshot)

        # Regex
        result = await assertion.to_have_text(
            r"Order #\\d+", screenshot, regex=True
        )
    """

    def __init__(
        self,
        locator: "BaseLocator | None" = None,
        config: "VisionConfig | None" = None,
        environment: "GUIEnvironment | None" = None,
    ) -> None:
        """Initialize text assertion.

        Args:
            locator: Optional locator for targeting region.
            config: Vision configuration.
            environment: GUI environment for typography hints.
        """
        self._locator = locator
        self._config = config
        self._environment = environment
        self._ocr_engine: OCREngine | None = None

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

    async def _get_region_from_locator(
        self,
        screenshot: NDArray[np.uint8],
    ) -> BoundingBox | None:
        """Get region from locator match.

        Args:
            screenshot: Screenshot to search.

        Returns:
            Bounding box of first match, or None.
        """
        if self._locator is None:
            return None

        match = await self._locator.find(screenshot)
        if match is not None:
            return match.bounds
        return None

    async def _extract_text(
        self,
        screenshot: NDArray[np.uint8],
        region: BoundingBox | None = None,
    ) -> tuple[str, list[Any]]:
        """Extract text from screenshot or region.

        Args:
            screenshot: Screenshot to process.
            region: Optional region to limit OCR.

        Returns:
            Tuple of (combined_text, ocr_results).
        """
        ocr = self._get_ocr_engine()
        results = await ocr.detect_text(screenshot, region)

        # Sort by position and combine
        results.sort(key=lambda r: (r.bounds.y, r.bounds.x))
        text = " ".join(r.text for r in results)

        return text, results

    def _text_matches(
        self,
        actual: str,
        expected: str,
        exact: bool,
        case_sensitive: bool,
        regex: bool,
    ) -> bool:
        """Check if text matches expected.

        Args:
            actual: Actual text from OCR.
            expected: Expected text pattern.
            exact: Require exact match.
            case_sensitive: Case-sensitive matching.
            regex: Treat expected as regex pattern.

        Returns:
            True if matches.
        """
        if regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(expected, flags)
            return pattern.search(actual) is not None

        compare_actual = actual if case_sensitive else actual.lower()
        compare_expected = expected if case_sensitive else expected.lower()

        if exact:
            return compare_actual.strip() == compare_expected.strip()
        else:
            return compare_expected in compare_actual

    async def to_have_text(
        self,
        expected_text: str,
        screenshot: NDArray[np.uint8],
        screenshot_callback: Any | None = None,
        timeout_ms: int | None = None,
        region: BoundingBox | None = None,
        exact: bool = True,
        case_sensitive: bool = True,
        regex: bool = False,
    ) -> AssertionResult:
        """Assert element/region has exact text.

        Args:
            expected_text: Expected text content.
            screenshot: Current screenshot.
            screenshot_callback: Async function to get fresh screenshots.
            timeout_ms: Maximum wait time.
            region: Region to limit OCR (or use locator).
            exact: Require exact match.
            case_sensitive: Case-sensitive matching.
            regex: Treat expected_text as regex pattern.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        timeout = self._get_timeout(timeout_ms)
        poll_interval = self._get_poll_interval()

        start_time = time.monotonic()
        deadline = start_time + (timeout / 1000.0)
        attempts = 0
        last_actual_text = ""
        current_screenshot = screenshot

        while True:
            attempts += 1
            try:
                # Get region from locator if not provided
                search_region = region
                if search_region is None and self._locator is not None:
                    search_region = await self._get_region_from_locator(current_screenshot)

                # Extract text
                actual_text, ocr_results = await self._extract_text(
                    current_screenshot, search_region
                )
                last_actual_text = actual_text

                # Check match
                if self._text_matches(actual_text, expected_text, exact, case_sensitive, regex):
                    elapsed_ms = int((time.monotonic() - start_time) * 1000)

                    # Build match from region
                    best_match = None
                    if search_region is not None:
                        center_x = search_region.x + search_region.width // 2
                        center_y = search_region.y + search_region.height // 2
                        best_match = VisionLocatorMatch(
                            bounds=search_region,
                            confidence=1.0,
                            center=(center_x, center_y),
                            text=actual_text,
                            locator_type=LocatorType.TEXT,
                        )
                    elif ocr_results:
                        # Use bounds of first result
                        first = ocr_results[0]
                        center_x = first.bounds.x + first.bounds.width // 2
                        center_y = first.bounds.y + first.bounds.height // 2
                        best_match = VisionLocatorMatch(
                            bounds=first.bounds,
                            confidence=first.confidence,
                            center=(center_x, center_y),
                            text=actual_text,
                            locator_type=LocatorType.TEXT,
                        )

                    completed_at = utc_now()
                    return AssertionResult(
                        assertion_id="text_have_text",
                        assertion_method="to_have_text",
                        status=AssertionStatus.PASSED,
                        started_at=started_at,
                        completed_at=completed_at,
                        expected_value=expected_text,
                        actual_value=actual_text,
                        matches_found=len(ocr_results),
                        best_match=best_match,
                        duration_ms=elapsed_ms,
                        retry_count=attempts - 1,
                    )

            except Exception as e:
                logger.debug(f"Text assertion failed: {e}")

            # Check timeout
            now = time.monotonic()
            if now >= deadline:
                elapsed_ms = int((now - start_time) * 1000)

                completed_at = utc_now()
                return AssertionResult(
                    assertion_id="text_have_text",
                    assertion_method="to_have_text",
                    status=AssertionStatus.FAILED,
                    started_at=started_at,
                    completed_at=completed_at,
                    error_message=f"Text mismatch: expected '{expected_text}', got '{last_actual_text}'",
                    expected_value=expected_text,
                    actual_value=last_actual_text,
                    matches_found=0,
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

    async def to_contain_text(
        self,
        expected_text: str,
        screenshot: NDArray[np.uint8],
        screenshot_callback: Any | None = None,
        timeout_ms: int | None = None,
        region: BoundingBox | None = None,
        case_sensitive: bool = True,
    ) -> AssertionResult:
        """Assert element/region contains text.

        Convenience method that calls to_have_text with exact=False.

        Args:
            expected_text: Text to search for.
            screenshot: Current screenshot.
            screenshot_callback: Async function to get fresh screenshots.
            timeout_ms: Maximum wait time.
            region: Region to limit OCR.
            case_sensitive: Case-sensitive matching.

        Returns:
            Assertion result.
        """
        result = await self.to_have_text(
            expected_text=expected_text,
            screenshot=screenshot,
            screenshot_callback=screenshot_callback,
            timeout_ms=timeout_ms,
            region=region,
            exact=False,
            case_sensitive=case_sensitive,
            regex=False,
        )

        # Update method name in result
        result.assertion_method = "to_contain_text"
        result.assertion_id = "text_contain_text"

        return result

    async def to_match_text(
        self,
        pattern: str,
        screenshot: NDArray[np.uint8],
        screenshot_callback: Any | None = None,
        timeout_ms: int | None = None,
        region: BoundingBox | None = None,
        case_sensitive: bool = True,
    ) -> AssertionResult:
        """Assert element/region text matches regex pattern.

        Convenience method that calls to_have_text with regex=True.

        Args:
            pattern: Regex pattern to match.
            screenshot: Current screenshot.
            screenshot_callback: Async function to get fresh screenshots.
            timeout_ms: Maximum wait time.
            region: Region to limit OCR.
            case_sensitive: Case-sensitive matching.

        Returns:
            Assertion result.
        """
        result = await self.to_have_text(
            expected_text=pattern,
            screenshot=screenshot,
            screenshot_callback=screenshot_callback,
            timeout_ms=timeout_ms,
            region=region,
            exact=False,
            case_sensitive=case_sensitive,
            regex=True,
        )

        # Update method name in result
        result.assertion_method = "to_match_text"
        result.assertion_id = "text_match_text"

        return result

    async def to_have_value(
        self,
        expected_value: str,
        screenshot: NDArray[np.uint8],
        screenshot_callback: Any | None = None,
        timeout_ms: int | None = None,
        region: BoundingBox | None = None,
    ) -> AssertionResult:
        """Assert input field has value.

        For vision-based testing, this is equivalent to to_have_text
        since we can only see the displayed value.

        Args:
            expected_value: Expected input value.
            screenshot: Current screenshot.
            screenshot_callback: Async function to get fresh screenshots.
            timeout_ms: Maximum wait time.
            region: Region of the input field.

        Returns:
            Assertion result.
        """
        result = await self.to_have_text(
            expected_text=expected_value,
            screenshot=screenshot,
            screenshot_callback=screenshot_callback,
            timeout_ms=timeout_ms,
            region=region,
            exact=True,
            case_sensitive=True,
            regex=False,
        )

        # Update method name in result
        result.assertion_method = "to_have_value"
        result.assertion_id = "text_have_value"

        return result


__all__ = ["TextAssertion"]
