"""
Verification pipeline for extracted clickable elements.

Uses qontinui's pattern matching to verify that extracted element
screenshots can be reliably detected in page screenshots.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image as PILImage

from qontinui.find.filters import NMSFilter, SimilarityFilter
from qontinui.find.find_executor import FindExecutor
from qontinui.find.matchers import TemplateMatcher
from qontinui.find.screenshot import ScreenshotProvider
from qontinui.model.element import Image, Pattern, Region

logger = logging.getLogger(__name__)


class StaticScreenshotProvider(ScreenshotProvider):
    """Screenshot provider that returns a static pre-loaded image.

    Used for verification of extracted elements against page screenshots.
    """

    def __init__(self, image: PILImage.Image) -> None:
        """Initialize with a static screenshot image.

        Args:
            image: PIL Image to return when capture() is called
        """
        self._image = image

    def capture(self, region: Region | None = None) -> PILImage.Image:
        """Return the static screenshot (optionally cropped to region).

        Args:
            region: Optional region to crop to. If None, returns full image.

        Returns:
            PIL Image of the screenshot (full or cropped).
        """
        if region is None:
            return self._image

        # Crop to the specified region
        return self._image.crop(
            (region.x, region.y, region.x + region.width, region.y + region.height)
        )


@dataclass
class VerificationResult:
    """Result of verifying a single extracted element."""

    element_id: str
    is_verified: bool
    match_confidence: float
    match_location: tuple[int, int] | None = None
    expected_location: tuple[int, int] | None = None
    location_error: float | None = None  # Pixels difference from expected
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "is_verified": self.is_verified,
            "match_confidence": self.match_confidence,
            "match_location": self.match_location,
            "expected_location": self.expected_location,
            "location_error": self.location_error,
            "error_message": self.error_message,
        }


@dataclass
class VerificationMetrics:
    """Aggregate metrics from verification run."""

    total: int = 0
    verified: int = 0
    failed: int = 0
    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    avg_location_error: float = 0.0
    verification_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "verified": self.verified,
            "failed": self.failed,
            "avg_confidence": self.avg_confidence,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "avg_location_error": self.avg_location_error,
            "verification_rate": self.verification_rate,
        }


@dataclass
class ExtractedClickable:
    """Represents an extracted clickable element with its data."""

    element_id: str
    selector: str
    tag_name: str
    text: str | None = None
    aria_label: str | None = None
    bounding_box: dict[str, int] = field(default_factory=dict)  # x, y, width, height
    screenshot: np.ndarray | None = None  # Element screenshot
    page_screenshot_before: np.ndarray | None = None
    page_screenshot_after: np.ndarray | None = None

    # Verification results (filled by verification step)
    is_verified: bool = False
    match_confidence: float = 0.0
    match_location: tuple[int, int] | None = None

    # Risk assessment
    risk_level: str = "unknown"
    risk_reason: str = ""

    # Click status
    was_clicked: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes large numpy arrays)."""
        return {
            "element_id": self.element_id,
            "selector": self.selector,
            "tag_name": self.tag_name,
            "text": self.text,
            "aria_label": self.aria_label,
            "bounding_box": self.bounding_box,
            "is_verified": self.is_verified,
            "match_confidence": self.match_confidence,
            "match_location": self.match_location,
            "risk_level": self.risk_level,
            "risk_reason": self.risk_reason,
            "was_clicked": self.was_clicked,
            "error": self.error,
            "has_screenshot": self.screenshot is not None,
            "has_page_screenshot_before": self.page_screenshot_before is not None,
            "has_page_screenshot_after": self.page_screenshot_after is not None,
        }


class ClickableVerifier:
    """Verify extracted clickables are reliably detectable using qontinui pattern matching."""

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        location_tolerance: int = 15,
    ):
        """Initialize verifier.

        Args:
            similarity_threshold: Minimum similarity score to consider verified
            location_tolerance: Maximum pixel distance for location matching
        """
        self.similarity_threshold = similarity_threshold
        self.location_tolerance = location_tolerance

    def verify_clickable(
        self,
        clickable: ExtractedClickable,
    ) -> VerificationResult:
        """Verify an element screenshot can be found in its page screenshot.

        Args:
            clickable: ExtractedClickable with screenshot and page_screenshot_before

        Returns:
            VerificationResult with match details
        """
        if clickable.screenshot is None:
            return VerificationResult(
                element_id=clickable.element_id,
                is_verified=False,
                match_confidence=0.0,
                error_message="No element screenshot available",
            )

        if clickable.page_screenshot_before is None:
            return VerificationResult(
                element_id=clickable.element_id,
                is_verified=False,
                match_confidence=0.0,
                error_message="No page screenshot available",
            )

        try:
            # Convert numpy arrays to PIL Images
            element_pil = PILImage.fromarray(clickable.screenshot)
            page_pil = PILImage.fromarray(clickable.page_screenshot_before)

            # Create pattern from element image
            element_image = Image.from_pil(element_pil, name="element")
            pattern = Pattern.from_image(element_image)
            pattern.similarity = self.similarity_threshold

            # Create screenshot provider
            screenshot_provider = StaticScreenshotProvider(page_pil)

            # Configure template matcher
            matcher = TemplateMatcher(
                method="TM_CCOEFF_NORMED",
                nms_overlap_threshold=0.3,
            )

            # Configure filters
            filters = [
                SimilarityFilter(min_similarity=self.similarity_threshold),
                NMSFilter(iou_threshold=0.3),
            ]

            # Create executor
            executor = FindExecutor(
                screenshot_provider=screenshot_provider,
                matcher=matcher,
                filters=filters,
            )

            # Execute find operation
            matches = executor.execute(
                pattern=pattern,
                similarity=self.similarity_threshold,
                find_all=False,
            )

            if matches and len(matches) > 0:
                match = matches[0]
                match_x = match.region.x if hasattr(match, "region") else 0
                match_y = match.region.y if hasattr(match, "region") else 0
                confidence = match.similarity if hasattr(match, "similarity") else 0.0

                # Calculate expected location from bounding box
                expected_x = clickable.bounding_box.get("x", 0)
                expected_y = clickable.bounding_box.get("y", 0)

                # Calculate location error
                location_error = ((match_x - expected_x) ** 2 + (match_y - expected_y) ** 2) ** 0.5

                # Verify both confidence and location
                location_ok = location_error <= self.location_tolerance
                confidence_ok = confidence >= self.similarity_threshold

                is_verified = location_ok and confidence_ok

                # Update clickable object
                clickable.is_verified = is_verified
                clickable.match_confidence = confidence
                clickable.match_location = (match_x, match_y)

                return VerificationResult(
                    element_id=clickable.element_id,
                    is_verified=is_verified,
                    match_confidence=confidence,
                    match_location=(match_x, match_y),
                    expected_location=(expected_x, expected_y),
                    location_error=location_error,
                    error_message=(
                        None
                        if is_verified
                        else (
                            f"Location error: {location_error:.1f}px"
                            if not location_ok
                            else f"Low confidence: {confidence:.2f}"
                        )
                    ),
                )
            else:
                # No match found
                clickable.is_verified = False
                clickable.match_confidence = 0.0

                return VerificationResult(
                    element_id=clickable.element_id,
                    is_verified=False,
                    match_confidence=0.0,
                    error_message="Pattern not found in page screenshot",
                )

        except Exception as e:
            logger.warning(f"Verification failed for {clickable.element_id}: {e}")
            return VerificationResult(
                element_id=clickable.element_id,
                is_verified=False,
                match_confidence=0.0,
                error_message=str(e),
            )

    def verify_all(
        self,
        clickables: list[ExtractedClickable],
    ) -> tuple[list[ExtractedClickable], VerificationMetrics]:
        """Verify all clickables and return metrics.

        Args:
            clickables: List of extracted clickables to verify

        Returns:
            Tuple of (updated clickables, metrics)
        """
        results: list[VerificationResult] = []
        confidences: list[float] = []
        location_errors: list[float] = []

        for clickable in clickables:
            result = self.verify_clickable(clickable)
            results.append(result)

            if result.is_verified:
                confidences.append(result.match_confidence)
                if result.location_error is not None:
                    location_errors.append(result.location_error)

        # Calculate metrics
        verified_count = sum(1 for r in results if r.is_verified)
        failed_count = len(results) - verified_count

        metrics = VerificationMetrics(
            total=len(clickables),
            verified=verified_count,
            failed=failed_count,
            avg_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            min_confidence=min(confidences) if confidences else 0.0,
            max_confidence=max(confidences) if confidences else 0.0,
            avg_location_error=(
                sum(location_errors) / len(location_errors) if location_errors else 0.0
            ),
            verification_rate=verified_count / len(clickables) if clickables else 0.0,
        )

        return clickables, metrics


class BatchVerifier:
    """Batch verification with progress reporting."""

    def __init__(
        self,
        verifier: ClickableVerifier | None = None,
        on_progress: Any = None,  # Callable[[str, int], None]
    ):
        """Initialize batch verifier.

        Args:
            verifier: ClickableVerifier instance (creates default if None)
            on_progress: Callback for progress updates (stage, percent)
        """
        self.verifier = verifier or ClickableVerifier()
        self.on_progress = on_progress

    def verify_batch(
        self,
        clickables: list[ExtractedClickable],
        batch_size: int = 10,
    ) -> tuple[list[ExtractedClickable], VerificationMetrics]:
        """Verify clickables in batches with progress reporting.

        Args:
            clickables: List of clickables to verify
            batch_size: Number of elements per batch

        Returns:
            Tuple of (verified clickables, metrics)
        """
        total = len(clickables)
        all_confidences: list[float] = []
        all_location_errors: list[float] = []
        verified_count = 0

        for i in range(0, total, batch_size):
            batch = clickables[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size

            # Report progress
            if self.on_progress:
                percent = int((i / total) * 100)
                self.on_progress(f"Verifying batch {batch_num}/{total_batches}", percent)

            # Verify batch
            for clickable in batch:
                result = self.verifier.verify_clickable(clickable)

                if result.is_verified:
                    verified_count += 1
                    all_confidences.append(result.match_confidence)
                    if result.location_error is not None:
                        all_location_errors.append(result.location_error)

        # Report completion
        if self.on_progress:
            self.on_progress("Verification complete", 100)

        # Calculate final metrics
        failed_count = total - verified_count
        metrics = VerificationMetrics(
            total=total,
            verified=verified_count,
            failed=failed_count,
            avg_confidence=(
                sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            ),
            min_confidence=min(all_confidences) if all_confidences else 0.0,
            max_confidence=max(all_confidences) if all_confidences else 0.0,
            avg_location_error=(
                sum(all_location_errors) / len(all_location_errors) if all_location_errors else 0.0
            ),
            verification_rate=verified_count / total if total else 0.0,
        )

        return clickables, metrics
