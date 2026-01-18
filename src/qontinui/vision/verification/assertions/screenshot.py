"""Screenshot comparison assertions for visual regression testing.

Provides toMatchScreenshot assertion with multiple comparison methods
and intelligent masking based on discovered GUI environment.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

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
    from qontinui_schemas.testing.environment import GUIEnvironment

    from qontinui.vision.verification.config import VisionConfig

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of screenshot comparison."""

    matches: bool
    similarity_score: float
    method: str
    diff_regions: list[BoundingBox] = field(default_factory=list)
    diff_image: NDArray[np.uint8] | None = None
    masked_regions: list[BoundingBox] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ScreenshotComparator:
    """Screenshot comparison engine with multiple methods.

    Supports:
    - Pixel-by-pixel comparison
    - SSIM (Structural Similarity Index)
    - Perceptual hashing
    - Feature-based comparison

    Integrates with GUI environment for intelligent masking
    of dynamic content regions.

    Usage:
        comparator = ScreenshotComparator(config, environment)

        # Compare screenshots
        result = comparator.compare(actual, expected)

        # With custom threshold
        result = comparator.compare(actual, expected, threshold=0.98)

        # With specific method
        result = comparator.compare(actual, expected, method="ssim")
    """

    def __init__(
        self,
        config: "VisionConfig | None" = None,
        environment: "GUIEnvironment | None" = None,
    ) -> None:
        """Initialize comparator.

        Args:
            config: Vision configuration.
            environment: GUI environment for auto-masking.
        """
        self._config = config
        self._environment = environment

    def _get_threshold(self, threshold: float | None) -> float:
        """Get comparison threshold.

        Args:
            threshold: Override threshold.

        Returns:
            Threshold value (0.0-1.0).
        """
        if threshold is not None:
            return threshold
        if self._config is not None:
            return self._config.comparison.default_threshold
        return 0.95

    def _get_method(self, method: str | None) -> str:
        """Get comparison method.

        Args:
            method: Override method.

        Returns:
            Method name.
        """
        if method is not None:
            return method
        if self._config is not None:
            return self._config.comparison.default_method
        return "ssim"

    def compare(
        self,
        actual: NDArray[np.uint8],
        expected: NDArray[np.uint8],
        threshold: float | None = None,
        method: str | None = None,
        mask_regions: list[BoundingBox] | None = None,
        auto_mask: bool = True,
    ) -> ComparisonResult:
        """Compare two screenshots.

        Args:
            actual: Actual screenshot.
            expected: Expected/baseline screenshot.
            threshold: Similarity threshold (0.0-1.0).
            method: Comparison method ('pixel', 'ssim', 'phash', 'feature').
            mask_regions: Regions to ignore during comparison.
            auto_mask: Automatically mask dynamic regions from environment.

        Returns:
            Comparison result.
        """
        threshold = self._get_threshold(threshold)
        method = self._get_method(method)

        # Collect mask regions
        all_masks: list[BoundingBox] = []
        if mask_regions:
            all_masks.extend(mask_regions)

        if auto_mask and self._environment is not None:
            auto_masks = self._get_auto_mask_regions()
            all_masks.extend(auto_masks)

        # Apply masks
        actual_masked = self._apply_masks(actual.copy(), all_masks)
        expected_masked = self._apply_masks(expected.copy(), all_masks)

        # Resize if needed
        if actual_masked.shape != expected_masked.shape:
            resized = cv2.resize(
                expected_masked,
                (actual_masked.shape[1], actual_masked.shape[0]),
            )
            expected_masked = cast(NDArray[np.uint8], resized)

        # Run comparison
        if method == "pixel":
            result = self._compare_pixel(actual_masked, expected_masked, threshold)
        elif method == "ssim":
            result = self._compare_ssim(actual_masked, expected_masked, threshold)
        elif method == "phash":
            result = self._compare_phash(actual_masked, expected_masked, threshold)
        elif method == "feature":
            result = self._compare_feature(actual_masked, expected_masked, threshold)
        else:
            logger.warning(f"Unknown method {method}, falling back to ssim")
            result = self._compare_ssim(actual_masked, expected_masked, threshold)

        result.masked_regions = all_masks
        return result

    def _get_auto_mask_regions(self) -> list[BoundingBox]:
        """Get regions to auto-mask from environment.

        Returns:
            List of regions to mask.
        """
        masks: list[BoundingBox] = []

        if self._environment is None:
            return masks

        # Mask dynamic text regions (timestamps, counters, etc.)
        if self._environment.typography:
            for region in self._environment.typography.common_text_regions:
                # Check if region is marked as dynamic
                # Access role attribute safely as it may not exist on all schema versions
                region_role = getattr(region, "role", None)
                if region_role in ("timestamp", "counter", "status", "notification"):
                    # Convert from environment BoundingBox to assertions BoundingBox
                    masks.append(
                        BoundingBox(
                            x=region.bounds.x,
                            y=region.bounds.y,
                            width=region.bounds.width,
                            height=region.bounds.height,
                        )
                    )

        return masks

    def _apply_masks(
        self,
        image: NDArray[np.uint8],
        masks: list[BoundingBox],
    ) -> NDArray[np.uint8]:
        """Apply masks to image.

        Args:
            image: Image to mask.
            masks: Regions to fill with solid color.

        Returns:
            Masked image.
        """
        for mask in masks:
            x1 = max(0, mask.x)
            y1 = max(0, mask.y)
            x2 = min(image.shape[1], mask.x + mask.width)
            y2 = min(image.shape[0], mask.y + mask.height)

            if x2 > x1 and y2 > y1:
                # Fill with gray to neutralize
                image[y1:y2, x1:x2] = 128

        return image

    def _compare_pixel(
        self,
        actual: NDArray[np.uint8],
        expected: NDArray[np.uint8],
        threshold: float,
    ) -> ComparisonResult:
        """Pixel-by-pixel comparison.

        Args:
            actual: Actual image.
            expected: Expected image.
            threshold: Match threshold.

        Returns:
            Comparison result.
        """
        # Calculate absolute difference
        diff = cv2.absdiff(actual, expected)
        gray_diff: NDArray[np.uint8]
        if len(diff.shape) == 3:
            gray_diff = np.asarray(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
        else:
            gray_diff = np.asarray(diff, dtype=np.uint8)

        # Count different pixels
        _, thresh_result = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)
        thresh_uint8: NDArray[np.uint8] = np.asarray(thresh_result, dtype=np.uint8)
        diff_pixels = np.count_nonzero(thresh_uint8)
        total_pixels = thresh_uint8.size

        similarity = 1.0 - (diff_pixels / total_pixels)

        # Find diff regions using contours
        diff_regions = self._find_diff_regions(thresh_uint8)

        # Create diff visualization
        diff_image = self._create_diff_image(actual, expected, thresh_uint8)

        return ComparisonResult(
            matches=similarity >= threshold,
            similarity_score=similarity,
            method="pixel",
            diff_regions=diff_regions,
            diff_image=diff_image,
            metadata={
                "diff_pixels": diff_pixels,
                "total_pixels": total_pixels,
            },
        )

    def _compare_ssim(
        self,
        actual: NDArray[np.uint8],
        expected: NDArray[np.uint8],
        threshold: float,
    ) -> ComparisonResult:
        """SSIM (Structural Similarity Index) comparison.

        Args:
            actual: Actual image.
            expected: Expected image.
            threshold: Match threshold.

        Returns:
            Comparison result.
        """
        # Convert to grayscale
        actual_gray: NDArray[np.uint8]
        expected_gray: NDArray[np.uint8]
        if len(actual.shape) == 3:
            actual_gray = np.asarray(cv2.cvtColor(actual, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
            expected_gray = np.asarray(cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
        else:
            actual_gray = actual
            expected_gray = expected

        # Calculate SSIM
        try:
            from skimage.metrics import structural_similarity

            score, diff_map = structural_similarity(
                actual_gray,
                expected_gray,
                full=True,
                data_range=255,
            )

            # Convert diff map to uint8
            diff_map_uint8 = (255 - diff_map * 255).astype(np.uint8)
            _, thresh = cv2.threshold(diff_map_uint8, 50, 255, cv2.THRESH_BINARY)

        except ImportError:
            # Fall back to manual SSIM approximation
            score, thresh_result = self._manual_ssim(actual_gray, expected_gray)
            diff_map_uint8 = thresh_result

        thresh_uint8: NDArray[np.uint8] = np.asarray(diff_map_uint8, dtype=np.uint8)
        diff_regions = self._find_diff_regions(thresh_uint8)
        diff_image = self._create_diff_image(actual, expected, thresh_uint8)

        return ComparisonResult(
            matches=score >= threshold,
            similarity_score=float(score),
            method="ssim",
            diff_regions=diff_regions,
            diff_image=diff_image,
        )

    def _manual_ssim(
        self,
        actual: NDArray[np.uint8],
        expected: NDArray[np.uint8],
    ) -> tuple[float, NDArray[np.uint8]]:
        """Manual SSIM approximation when skimage not available.

        Args:
            actual: Actual grayscale image.
            expected: Expected grayscale image.

        Returns:
            Tuple of (score, diff_threshold).
        """
        # Constants for SSIM
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        actual_f = actual.astype(np.float64)
        expected_f = expected.astype(np.float64)

        # Means
        mu1 = cv2.GaussianBlur(actual_f, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(expected_f, (11, 11), 1.5)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        # Variances
        sigma1_sq = cv2.GaussianBlur(actual_f**2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(expected_f**2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(actual_f * expected_f, (11, 11), 1.5) - mu1_mu2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        score = float(ssim_map.mean())

        # Create diff threshold
        diff = cv2.absdiff(actual, expected)
        _, thresh_result = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        thresh: NDArray[np.uint8] = np.asarray(thresh_result, dtype=np.uint8)

        return score, thresh

    def _compare_phash(
        self,
        actual: NDArray[np.uint8],
        expected: NDArray[np.uint8],
        threshold: float,
    ) -> ComparisonResult:
        """Perceptual hash comparison.

        Args:
            actual: Actual image.
            expected: Expected image.
            threshold: Match threshold.

        Returns:
            Comparison result.
        """
        # Compute perceptual hashes
        actual_hash = self._compute_phash(actual)
        expected_hash = self._compute_phash(expected)

        # Calculate Hamming distance
        hamming_distance = bin(actual_hash ^ expected_hash).count("1")
        max_distance = 64  # 8x8 hash

        similarity = 1.0 - (hamming_distance / max_distance)

        # For diff visualization, fall back to pixel diff
        diff = cv2.absdiff(actual, expected)
        gray_diff_phash: NDArray[np.uint8]
        if len(diff.shape) == 3:
            gray_diff_phash = np.asarray(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
        else:
            gray_diff_phash = np.asarray(diff, dtype=np.uint8)
        _, thresh_result = cv2.threshold(gray_diff_phash, 10, 255, cv2.THRESH_BINARY)

        thresh_uint8: NDArray[np.uint8] = np.asarray(thresh_result, dtype=np.uint8)
        diff_regions = self._find_diff_regions(thresh_uint8)
        diff_image = self._create_diff_image(actual, expected, thresh_uint8)

        return ComparisonResult(
            matches=similarity >= threshold,
            similarity_score=similarity,
            method="phash",
            diff_regions=diff_regions,
            diff_image=diff_image,
            metadata={
                "hamming_distance": hamming_distance,
                "actual_hash": hex(actual_hash),
                "expected_hash": hex(expected_hash),
            },
        )

    def _compute_phash(self, image: NDArray[np.uint8], hash_size: int = 8) -> int:
        """Compute perceptual hash of image.

        Args:
            image: Input image.
            hash_size: Size of hash grid.

        Returns:
            Hash as integer.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize to hash_size + 1 for DCT
        resized = cv2.resize(gray, (hash_size + 1, hash_size))

        # Compute difference
        diff = resized[:, 1:] > resized[:, :-1]

        # Convert to hash
        hash_value = 0
        for i, bit in enumerate(diff.flatten()):
            if bit:
                hash_value |= 1 << i

        return hash_value

    def _compare_feature(
        self,
        actual: NDArray[np.uint8],
        expected: NDArray[np.uint8],
        threshold: float,
    ) -> ComparisonResult:
        """Feature-based comparison using ORB.

        Args:
            actual: Actual image.
            expected: Expected image.
            threshold: Match threshold.

        Returns:
            Comparison result.
        """
        # Convert to grayscale
        actual_gray_feature: NDArray[np.uint8]
        expected_gray_feature: NDArray[np.uint8]
        if len(actual.shape) == 3:
            actual_gray_feature = np.asarray(
                cv2.cvtColor(actual, cv2.COLOR_BGR2GRAY), dtype=np.uint8
            )
            expected_gray_feature = np.asarray(
                cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY), dtype=np.uint8
            )
        else:
            actual_gray_feature = actual
            expected_gray_feature = expected

        # Detect ORB features
        orb = cv2.ORB.create(nfeatures=500)

        # Create empty mask arrays for detectAndCompute (None is not accepted by mypy)
        empty_mask: NDArray[np.uint8] = np.array([], dtype=np.uint8)
        kp1, desc1 = orb.detectAndCompute(actual_gray_feature, empty_mask)
        kp2, desc2 = orb.detectAndCompute(expected_gray_feature, empty_mask)

        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            # No features found, fall back to pixel comparison
            return self._compare_pixel(actual, expected, threshold)

        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)

        # Calculate similarity based on good matches
        good_matches = [m for m in matches if m.distance < 50]
        max_matches = min(len(kp1), len(kp2))
        similarity = len(good_matches) / max_matches if max_matches > 0 else 0

        # For diff visualization
        diff = cv2.absdiff(actual, expected)
        gray_diff_feature: NDArray[np.uint8]
        if len(diff.shape) == 3:
            gray_diff_feature = np.asarray(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
        else:
            gray_diff_feature = np.asarray(diff, dtype=np.uint8)
        _, thresh_result = cv2.threshold(gray_diff_feature, 10, 255, cv2.THRESH_BINARY)

        thresh_uint8: NDArray[np.uint8] = np.asarray(thresh_result, dtype=np.uint8)
        diff_regions = self._find_diff_regions(thresh_uint8)
        diff_image = self._create_diff_image(actual, expected, thresh_uint8)

        return ComparisonResult(
            matches=similarity >= threshold,
            similarity_score=similarity,
            method="feature",
            diff_regions=diff_regions,
            diff_image=diff_image,
            metadata={
                "total_features_actual": len(kp1),
                "total_features_expected": len(kp2),
                "good_matches": len(good_matches),
            },
        )

    def _find_diff_regions(
        self,
        thresh: NDArray[np.uint8],
    ) -> list[BoundingBox]:
        """Find diff regions from threshold image.

        Args:
            thresh: Binary threshold image.

        Returns:
            List of bounding boxes for diff regions.
        """
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter small noise
            if w * h > 100:
                regions.append(BoundingBox(x=x, y=y, width=w, height=h))

        return regions

    def _create_diff_image(
        self,
        actual: NDArray[np.uint8],
        expected: NDArray[np.uint8],
        thresh: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Create visualization of differences.

        Args:
            actual: Actual image.
            expected: Expected image.
            thresh: Threshold image highlighting differences.

        Returns:
            Diff visualization image.
        """
        # Create side-by-side with diff overlay
        h, w = actual.shape[:2]

        # Create output canvas (3 images side by side)
        output = np.zeros((h, w * 3, 3), dtype=np.uint8)

        # Expected
        if len(expected.shape) == 2:
            output[:, :w] = cv2.cvtColor(expected, cv2.COLOR_GRAY2BGR)
        else:
            output[:, :w] = expected

        # Actual
        if len(actual.shape) == 2:
            output[:, w : w * 2] = cv2.cvtColor(actual, cv2.COLOR_GRAY2BGR)
        else:
            output[:, w : w * 2] = actual

        # Diff overlay (actual with red highlights)
        if len(actual.shape) == 2:
            diff_overlay = cv2.cvtColor(actual, cv2.COLOR_GRAY2BGR)
        else:
            diff_overlay = actual.copy()

        # Highlight differences in red
        diff_mask = thresh > 0
        diff_overlay[diff_mask] = [0, 0, 255]

        output[:, w * 2 :] = diff_overlay

        return output


class ScreenshotAssertion:
    """Assertion for screenshot comparison.

    Usage:
        assertion = ScreenshotAssertion(config, environment)
        result = await assertion.to_match_screenshot(
            actual_screenshot,
            baseline_path="baseline.png",
        )
    """

    def __init__(
        self,
        config: "VisionConfig | None" = None,
        environment: "GUIEnvironment | None" = None,
    ) -> None:
        """Initialize screenshot assertion.

        Args:
            config: Vision configuration.
            environment: GUI environment for auto-masking.
        """
        self._config = config
        self._environment = environment
        self._comparator = ScreenshotComparator(config, environment)

    def _get_baseline_dir(self) -> Path:
        """Get baseline screenshots directory.

        Returns:
            Path to baseline directory.
        """
        if self._config is not None and self._config.screenshot.baseline_dir:
            return Path(self._config.screenshot.baseline_dir)
        return Path(".dev-logs/baselines")

    def _get_diff_dir(self) -> Path:
        """Get diff screenshots directory.

        Returns:
            Path to diff directory.
        """
        if self._config is not None and self._config.screenshot.diff_dir:
            return Path(self._config.screenshot.diff_dir)
        return Path(".dev-logs/diffs")

    async def to_match_screenshot(
        self,
        actual: NDArray[np.uint8],
        baseline_name: str | None = None,
        baseline_path: str | Path | None = None,
        baseline_image: NDArray[np.uint8] | None = None,
        threshold: float | None = None,
        method: str | None = None,
        mask_regions: list[BoundingBox] | None = None,
        auto_mask: bool = True,
        update_baseline: bool = False,
    ) -> AssertionResult:
        """Assert screenshot matches baseline.

        Args:
            actual: Actual screenshot.
            baseline_name: Name for baseline (auto-generates path).
            baseline_path: Explicit path to baseline image.
            baseline_image: Baseline image array (instead of file).
            threshold: Similarity threshold.
            method: Comparison method.
            mask_regions: Regions to ignore.
            auto_mask: Auto-mask dynamic regions.
            update_baseline: Update baseline if not found.

        Returns:
            Assertion result.
        """
        started_at = utc_now()
        start_time = time.monotonic()

        # Determine baseline
        expected: NDArray[np.uint8] | None = None

        baseline_path_resolved: Path | None = None
        if baseline_image is not None:
            expected = baseline_image
        elif baseline_path is not None:
            baseline_path_resolved = Path(baseline_path)
            if baseline_path_resolved.exists():
                loaded = cv2.imread(str(baseline_path_resolved))
                if loaded is not None:
                    expected = loaded.astype(np.uint8)
        elif baseline_name is not None:
            baseline_path_resolved = self._get_baseline_dir() / f"{baseline_name}.png"
            if baseline_path_resolved.exists():
                loaded = cv2.imread(str(baseline_path_resolved))
                if loaded is not None:
                    expected = loaded.astype(np.uint8)

        # Handle missing baseline
        if expected is None:
            if update_baseline and baseline_path_resolved is not None:
                # Create baseline
                baseline_path_resolved.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(baseline_path_resolved), actual)

                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                completed_at = utc_now()
                return AssertionResult(
                    assertion_id="screenshot_match",
                    assertion_method="to_match_screenshot",
                    status=AssertionStatus.PASSED,
                    started_at=started_at,
                    completed_at=completed_at,
                    expected_value="new baseline",
                    actual_value="baseline created",
                    matches_found=1,
                    duration_ms=elapsed_ms,
                )
            else:
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                completed_at = utc_now()
                return AssertionResult(
                    assertion_id="screenshot_match",
                    assertion_method="to_match_screenshot",
                    status=AssertionStatus.FAILED,
                    started_at=started_at,
                    completed_at=completed_at,
                    error_message="Baseline not found",
                    expected_value="baseline image",
                    actual_value="not found",
                    matches_found=0,
                    duration_ms=elapsed_ms,
                )

        # Compare screenshots
        result = self._comparator.compare(
            actual=actual,
            expected=expected,
            threshold=threshold,
            method=method,
            mask_regions=mask_regions,
            auto_mask=auto_mask,
        )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # Save diff image if comparison failed
        diff_path: str | None = None
        if not result.matches and result.diff_image is not None:
            diff_dir = self._get_diff_dir()
            diff_dir.mkdir(parents=True, exist_ok=True)

            diff_name = baseline_name or "screenshot"
            timestamp = int(time.time())
            diff_path = str(diff_dir / f"{diff_name}_diff_{timestamp}.png")
            cv2.imwrite(diff_path, result.diff_image)

        completed_at = utc_now()
        if result.matches:
            return AssertionResult(
                assertion_id="screenshot_match",
                assertion_method="to_match_screenshot",
                status=AssertionStatus.PASSED,
                started_at=started_at,
                completed_at=completed_at,
                expected_value="baseline",
                actual_value=f"{result.similarity_score:.1%} match",
                matches_found=1,
                duration_ms=elapsed_ms,
            )
        else:
            return AssertionResult(
                assertion_id="screenshot_match",
                assertion_method="to_match_screenshot",
                status=AssertionStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                error_message=f"Screenshot differs ({result.similarity_score:.1%} similarity, {len(result.diff_regions)} regions)",
                expected_value="baseline",
                actual_value=f"{result.similarity_score:.1%} match",
                matches_found=0,
                duration_ms=elapsed_ms,
                screenshot_path=diff_path,
            )


# Global comparator instance
_screenshot_comparator: ScreenshotComparator | None = None


def get_screenshot_comparator(
    config: "VisionConfig | None" = None,
    environment: "GUIEnvironment | None" = None,
) -> ScreenshotComparator:
    """Get the global screenshot comparator instance.

    Args:
        config: Optional vision configuration.
        environment: Optional GUI environment.

    Returns:
        ScreenshotComparator instance.
    """
    global _screenshot_comparator
    if _screenshot_comparator is None:
        _screenshot_comparator = ScreenshotComparator(config=config, environment=environment)
    return _screenshot_comparator


__all__ = [
    "ComparisonResult",
    "ScreenshotAssertion",
    "ScreenshotComparator",
    "get_screenshot_comparator",
]
