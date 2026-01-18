"""Visual validation for action success verification.

Validates that actions succeeded by comparing before/after screenshots.
Catches failures immediately rather than discovering them later.
"""

import logging
import time
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .validation_types import (
    ChangedRegion,
    ChangeType,
    ExpectedChange,
    ValidationResult,
    VisualDiff,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class VisualValidator:
    """Validate actions by comparing before/after screenshots.

    Provides immediate feedback on whether an action succeeded by
    detecting visual changes on screen.

    Features:
    - Compare overall screen change
    - Check for specific element appearance/disappearance
    - Monitor specific regions for changes
    - Detect changed regions for debugging

    Attributes:
        change_threshold: Pixel difference threshold (0-255).
        min_region_size: Minimum pixels for a changed region.
    """

    def __init__(
        self,
        change_threshold: int = 25,
        min_region_size: int = 100,
    ) -> None:
        """Initialize visual validator.

        Args:
            change_threshold: Pixel difference threshold to consider changed.
            min_region_size: Minimum area to report as a changed region.
        """
        self.change_threshold = change_threshold
        self.min_region_size = min_region_size

    def validate(
        self,
        pre_screenshot: np.ndarray,
        post_screenshot: np.ndarray,
        expected: ExpectedChange | None = None,
    ) -> ValidationResult:
        """Validate that an action produced expected change.

        Args:
            pre_screenshot: Screenshot before action (BGR numpy array).
            post_screenshot: Screenshot after action (BGR numpy array).
            expected: What change was expected. If None, validates any change.

        Returns:
            ValidationResult indicating success or failure.
        """
        start_time = time.time()

        # Default expectation: any change
        if expected is None:
            expected = ExpectedChange(type=ChangeType.ANY_CHANGE)

        # Compute visual diff
        diff = self._compute_diff(pre_screenshot, post_screenshot)

        # Validate based on expected change type
        if expected.type == ChangeType.ANY_CHANGE:
            result = self._validate_any_change(diff, expected)

        elif expected.type == ChangeType.NO_CHANGE:
            result = self._validate_no_change(diff, expected)

        elif expected.type == ChangeType.ELEMENT_APPEARS:
            result = self._validate_element_appears(pre_screenshot, post_screenshot, diff, expected)

        elif expected.type == ChangeType.ELEMENT_DISAPPEARS:
            result = self._validate_element_disappears(
                pre_screenshot, post_screenshot, diff, expected
            )

        elif expected.type == ChangeType.REGION_CHANGES:
            result = self._validate_region_changes(pre_screenshot, post_screenshot, diff, expected)

        else:
            result = ValidationResult(
                success=False,
                message=f"Unknown change type: {expected.type}",
            )

        # Add common fields
        result.diff = diff
        result.expected = expected
        result.actual_change_percentage = diff.change_percentage
        result.validation_time_ms = (time.time() - start_time) * 1000

        return result

    def compute_diff(
        self,
        pre_screenshot: np.ndarray,
        post_screenshot: np.ndarray,
    ) -> VisualDiff:
        """Compute visual diff between two screenshots.

        Public wrapper for _compute_diff for external use.

        Args:
            pre_screenshot: Before screenshot.
            post_screenshot: After screenshot.

        Returns:
            VisualDiff with change information.
        """
        return self._compute_diff(pre_screenshot, post_screenshot)

    def _compute_diff(
        self,
        pre_screenshot: np.ndarray,
        post_screenshot: np.ndarray,
    ) -> VisualDiff:
        """Compute visual diff between two screenshots.

        Args:
            pre_screenshot: Before screenshot.
            post_screenshot: After screenshot.

        Returns:
            VisualDiff with change information.
        """
        # Convert to grayscale for comparison
        if len(pre_screenshot.shape) == 3:
            gray_pre = cv2.cvtColor(pre_screenshot, cv2.COLOR_BGR2GRAY)
        else:
            gray_pre = pre_screenshot

        if len(post_screenshot.shape) == 3:
            gray_post = cv2.cvtColor(post_screenshot, cv2.COLOR_BGR2GRAY)
        else:
            gray_post = post_screenshot

        # Ensure same size
        if gray_pre.shape != gray_post.shape:
            # Resize to match (use smaller dimensions)
            h = min(gray_pre.shape[0], gray_post.shape[0])
            w = min(gray_pre.shape[1], gray_post.shape[1])
            gray_pre = gray_pre[:h, :w]
            gray_post = gray_post[:h, :w]

        # Compute absolute difference
        diff = cv2.absdiff(gray_pre, gray_post)

        # Threshold to identify changed pixels
        _, thresh = cv2.threshold(diff, self.change_threshold, 255, cv2.THRESH_BINARY)

        # Calculate statistics
        changed_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        change_percentage = (changed_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

        # Calculate mean intensity of changes
        mean_intensity = float(np.mean(diff[thresh > 0])) / 255.0 if changed_pixels > 0 else 0.0

        # Find changed regions using contours
        changed_regions = self._find_changed_regions(thresh, diff)

        return VisualDiff(
            change_percentage=change_percentage,
            changed_pixel_count=changed_pixels,
            total_pixel_count=total_pixels,
            changed_regions=changed_regions,
            mean_change_intensity=mean_intensity,
        )

    def _find_changed_regions(
        self,
        thresh: np.ndarray,
        diff: np.ndarray,
    ) -> list[ChangedRegion]:
        """Find contiguous changed regions.

        Args:
            thresh: Thresholded difference image.
            diff: Raw difference image.

        Returns:
            List of changed regions.
        """
        # Find contours of changed areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_region_size:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Calculate change intensity for this region
            region_diff = diff[y : y + h, x : x + w]
            intensity = float(np.mean(region_diff)) / 255.0

            regions.append(
                ChangedRegion(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    change_intensity=intensity,
                )
            )

        # Sort by area (largest first)
        regions.sort(key=lambda r: r.width * r.height, reverse=True)

        return regions

    def _validate_any_change(
        self,
        diff: VisualDiff,
        expected: ExpectedChange,
    ) -> ValidationResult:
        """Validate that something changed."""
        min_change = expected.min_any_change

        if diff.change_percentage >= min_change:
            return ValidationResult(
                success=True,
                message=f"Screen changed by {diff.change_percentage:.2f}%",
            )
        else:
            return ValidationResult(
                success=False,
                message=f"Insufficient change: {diff.change_percentage:.2f}% < {min_change}%",
            )

    def _validate_no_change(
        self,
        diff: VisualDiff,
        expected: ExpectedChange,
    ) -> ValidationResult:
        """Validate that nothing changed."""
        max_change = expected.min_any_change

        if diff.change_percentage <= max_change:
            return ValidationResult(
                success=True,
                message=f"Screen stable ({diff.change_percentage:.2f}% change)",
            )
        else:
            return ValidationResult(
                success=False,
                message=f"Unexpected change: {diff.change_percentage:.2f}% > {max_change}%",
            )

    def _validate_element_appears(
        self,
        pre_screenshot: np.ndarray,
        post_screenshot: np.ndarray,
        diff: VisualDiff,
        expected: ExpectedChange,
    ) -> ValidationResult:
        """Validate that expected element appeared."""
        if expected.pattern is None:
            return ValidationResult(
                success=False,
                message="No pattern specified for ELEMENT_APPEARS validation",
            )

        pattern = self._prepare_pattern(expected.pattern)

        # Check if pattern is in post but not in pre
        pre_match = self._find_pattern(pre_screenshot, pattern, expected.similarity_threshold)
        post_match = self._find_pattern(post_screenshot, pattern, expected.similarity_threshold)

        if post_match and not pre_match:
            return ValidationResult(
                success=True,
                message=f"Element appeared at ({post_match[0]}, {post_match[1]}) with similarity {post_match[2]:.3f}",
            )
        elif post_match and pre_match:
            return ValidationResult(
                success=False,
                message="Element was already visible before action",
            )
        else:
            best_match = post_match[2] if post_match else 0.0
            return ValidationResult(
                success=False,
                message=f"Element did not appear (best match: {best_match:.3f})",
            )

    def _validate_element_disappears(
        self,
        pre_screenshot: np.ndarray,
        post_screenshot: np.ndarray,
        diff: VisualDiff,
        expected: ExpectedChange,
    ) -> ValidationResult:
        """Validate that expected element disappeared."""
        if expected.pattern is None:
            return ValidationResult(
                success=False,
                message="No pattern specified for ELEMENT_DISAPPEARS validation",
            )

        pattern = self._prepare_pattern(expected.pattern)

        # Check if pattern was in pre but not in post
        pre_match = self._find_pattern(pre_screenshot, pattern, expected.similarity_threshold)
        post_match = self._find_pattern(post_screenshot, pattern, expected.similarity_threshold)

        if pre_match and not post_match:
            return ValidationResult(
                success=True,
                message="Element successfully disappeared",
            )
        elif not pre_match and not post_match:
            return ValidationResult(
                success=False,
                message="Element was not visible before action",
            )
        else:
            # post_match is guaranteed to exist in this branch (pre_match exists or doesn't, post_match exists)
            assert post_match is not None
            return ValidationResult(
                success=False,
                message=f"Element still visible at ({post_match[0]}, {post_match[1]})",
            )

    def _validate_region_changes(
        self,
        pre_screenshot: np.ndarray,
        post_screenshot: np.ndarray,
        diff: VisualDiff,
        expected: ExpectedChange,
    ) -> ValidationResult:
        """Validate that specific region changed."""
        if expected.region is None:
            return ValidationResult(
                success=False,
                message="No region specified for REGION_CHANGES validation",
            )

        x, y, w, h = expected.region

        # Extract region from both screenshots
        pre_region = pre_screenshot[y : y + h, x : x + w]
        post_region = post_screenshot[y : y + h, x : x + w]

        # Compute diff for just this region
        region_diff = self._compute_diff(pre_region, post_region)

        if region_diff.change_percentage >= expected.min_change_threshold:
            return ValidationResult(
                success=True,
                message=f"Region changed by {region_diff.change_percentage:.2f}%",
            )
        else:
            return ValidationResult(
                success=False,
                message=f"Region change {region_diff.change_percentage:.2f}% < threshold {expected.min_change_threshold}%",
            )

    def _prepare_pattern(self, pattern: Any) -> np.ndarray:
        """Prepare pattern for matching.

        Args:
            pattern: Pattern (numpy array or Pattern object).

        Returns:
            BGR numpy array.
        """
        # Handle Pattern object
        if hasattr(pattern, "pixel_data"):
            pattern = pattern.pixel_data

        # Remove alpha channel if present
        if len(pattern.shape) == 3 and pattern.shape[2] == 4:
            pattern = pattern[:, :, :3]

        return np.asarray(pattern)

    def _find_pattern(
        self,
        screenshot: np.ndarray,
        pattern: np.ndarray,
        threshold: float,
    ) -> tuple[int, int, float] | None:
        """Find pattern in screenshot.

        Args:
            screenshot: Screenshot to search.
            pattern: Pattern to find.
            threshold: Minimum similarity.

        Returns:
            (x, y, similarity) if found, None otherwise.
        """
        try:
            result = cv2.matchTemplate(screenshot, pattern, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val >= threshold:
                x = max_loc[0] + pattern.shape[1] // 2
                y = max_loc[1] + pattern.shape[0] // 2
                return (x, y, max_val)

            return None

        except cv2.error as e:
            logger.warning(f"Pattern matching error: {e}")
            return None


# Global validator instance
_default_validator: VisualValidator | None = None


def get_visual_validator() -> VisualValidator:
    """Get the global visual validator instance.

    Returns:
        VisualValidator instance.
    """
    global _default_validator
    if _default_validator is None:
        _default_validator = VisualValidator()
    return _default_validator


def set_visual_validator(validator: VisualValidator) -> None:
    """Set the global visual validator instance.

    Args:
        validator: VisualValidator to use globally.
    """
    global _default_validator
    _default_validator = validator
