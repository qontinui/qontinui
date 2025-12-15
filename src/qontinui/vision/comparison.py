"""Visual comparison algorithms for regression testing.

This module provides image comparison capabilities for detecting visual
differences between baseline screenshots and current state captures.

Algorithms:
- SSIM: Structural Similarity Index - detects layout/structural changes
- Pixel Diff: Pixel-by-pixel comparison with color tolerance
- Perceptual Hash: Layout-agnostic content comparison

Usage:
    from qontinui.vision.comparison import VisualComparator, ComparisonResult

    comparator = VisualComparator()
    result = comparator.compare_ssim(baseline_img, screenshot_img, threshold=0.95)

    if not result.passed:
        diff_png = comparator.generate_diff_image(baseline_img, screenshot_img, result.diff_mask)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np
from PIL import Image

# Optional imports with graceful fallback
try:
    from skimage.metrics import structural_similarity as ssim

    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False

try:
    import imagehash

    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False


class ComparisonAlgorithm(str, Enum):
    """Available comparison algorithms."""

    SSIM = "ssim"
    PIXEL_DIFF = "pixel_diff"
    PERCEPTUAL_HASH = "perceptual_hash"


@dataclass
class IgnoreRegion:
    """Region to ignore during comparison (for dynamic content)."""

    x: int
    y: int
    width: int
    height: int
    name: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IgnoreRegion:
        """Create from dictionary."""
        return cls(
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
            name=data.get("name"),
        )


@dataclass
class DiffRegion:
    """A region where differences were detected."""

    x: int
    y: int
    width: int
    height: int
    change_percentage: float  # 0.0 to 1.0 - how much of this region changed
    pixel_count: int  # Number of changed pixels in this region

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "change_percentage": self.change_percentage,
            "pixel_count": self.pixel_count,
        }


@dataclass
class ComparisonResult:
    """Result of visual comparison between two images."""

    similarity_score: float  # 0.0 to 1.0, where 1.0 is identical
    passed: bool  # True if similarity >= threshold
    threshold: float  # The threshold used for comparison
    algorithm: ComparisonAlgorithm
    execution_time_ms: int
    diff_regions: list[DiffRegion] = field(default_factory=list)
    diff_mask: np.ndarray | None = None  # Binary mask of differences
    error: str | None = None  # Error message if comparison failed

    def to_dict(self) -> dict:
        """Convert to dictionary representation (excludes diff_mask for serialization)."""
        return {
            "similarity_score": self.similarity_score,
            "passed": self.passed,
            "threshold": self.threshold,
            "algorithm": self.algorithm.value,
            "execution_time_ms": self.execution_time_ms,
            "diff_regions": [r.to_dict() for r in self.diff_regions],
            "error": self.error,
        }


class VisualComparator:
    """Visual comparison algorithms for regression testing.

    This class provides multiple image comparison algorithms optimized for
    detecting visual regressions in UI screenshots.

    Example:
        comparator = VisualComparator()

        # Compare using SSIM (best for structural changes)
        result = comparator.compare_ssim(baseline, screenshot, threshold=0.95)

        # Compare with ignore regions
        regions = [IgnoreRegion(x=10, y=10, width=100, height=50, name="timestamp")]
        masked_baseline = comparator.apply_ignore_mask(baseline, regions)
        masked_screenshot = comparator.apply_ignore_mask(screenshot, regions)
        result = comparator.compare_ssim(masked_baseline, masked_screenshot)

        # Generate diff visualization
        if not result.passed:
            diff_png = comparator.generate_diff_image(baseline, screenshot, result.diff_mask)
    """

    # Default thresholds for each algorithm
    DEFAULT_THRESHOLDS = {
        ComparisonAlgorithm.SSIM: 0.95,
        ComparisonAlgorithm.PIXEL_DIFF: 0.99,
        ComparisonAlgorithm.PERCEPTUAL_HASH: 0.90,
    }

    def compare(
        self,
        baseline: np.ndarray,
        screenshot: np.ndarray,
        algorithm: ComparisonAlgorithm | str = ComparisonAlgorithm.SSIM,
        threshold: float | None = None,
        ignore_regions: list[IgnoreRegion] | None = None,
        **kwargs,
    ) -> ComparisonResult:
        """Compare two images using the specified algorithm.

        Args:
            baseline: Baseline image (BGR or grayscale numpy array)
            screenshot: Current screenshot to compare (BGR or grayscale numpy array)
            algorithm: Comparison algorithm to use
            threshold: Similarity threshold (uses algorithm default if not specified)
            ignore_regions: Regions to mask out before comparison
            **kwargs: Additional algorithm-specific parameters

        Returns:
            ComparisonResult with similarity score and diff information
        """
        if isinstance(algorithm, str):
            algorithm = ComparisonAlgorithm(algorithm)

        # Apply ignore mask if regions specified
        if ignore_regions:
            baseline = self.apply_ignore_mask(baseline, ignore_regions)
            screenshot = self.apply_ignore_mask(screenshot, ignore_regions)

        # Use default threshold if not specified
        if threshold is None:
            threshold = self.DEFAULT_THRESHOLDS[algorithm]

        # Route to appropriate comparison method
        if algorithm == ComparisonAlgorithm.SSIM:
            return self.compare_ssim(baseline, screenshot, threshold, **kwargs)
        elif algorithm == ComparisonAlgorithm.PIXEL_DIFF:
            return self.compare_pixel_diff(baseline, screenshot, threshold, **kwargs)
        elif algorithm == ComparisonAlgorithm.PERCEPTUAL_HASH:
            return self.compare_perceptual_hash(baseline, screenshot, threshold, **kwargs)
        else:
            return ComparisonResult(
                similarity_score=0.0,
                passed=False,
                threshold=threshold,
                algorithm=algorithm,
                execution_time_ms=0,
                error=f"Unknown algorithm: {algorithm}",
            )

    def compare_ssim(
        self,
        baseline: np.ndarray,
        screenshot: np.ndarray,
        threshold: float = 0.95,
        win_size: int | None = None,
    ) -> ComparisonResult:
        """Compare images using Structural Similarity Index (SSIM).

        SSIM measures structural similarity between images, making it effective
        for detecting layout changes, text modifications, and structural shifts.
        It's less sensitive to minor color variations than pixel diff.

        Args:
            baseline: Baseline image (BGR or grayscale)
            screenshot: Screenshot to compare (BGR or grayscale)
            threshold: Minimum similarity score to pass (default 0.95)
            win_size: Window size for SSIM calculation (default auto)

        Returns:
            ComparisonResult with SSIM score and diff regions
        """
        start_time = time.perf_counter()

        if not SSIM_AVAILABLE:
            return ComparisonResult(
                similarity_score=0.0,
                passed=False,
                threshold=threshold,
                algorithm=ComparisonAlgorithm.SSIM,
                execution_time_ms=0,
                error="scikit-image not installed. Install with: pip install scikit-image",
            )

        try:
            # Ensure same size
            baseline, screenshot = self._resize_to_match(baseline, screenshot)

            # Convert to grayscale for SSIM
            baseline_gray = self._to_grayscale(baseline)
            screenshot_gray = self._to_grayscale(screenshot)

            # Determine window size
            if win_size is None:
                # Window size must be odd and <= image dimensions
                min_dim = min(baseline_gray.shape[:2])
                win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
                win_size = max(3, win_size)  # Minimum 3

            # Calculate SSIM with full diff image
            score, diff_image = ssim(
                baseline_gray,
                screenshot_gray,
                win_size=win_size,
                full=True,
                data_range=255,
            )

            # Create binary diff mask (areas below threshold)
            # diff_image is in range [0, 1], where 1 = identical
            diff_mask = ((1 - diff_image) * 255).astype(np.uint8)
            _, diff_mask = cv2.threshold(diff_mask, 25, 255, cv2.THRESH_BINARY)

            # Extract diff regions from mask
            diff_regions = self._extract_diff_regions(diff_mask)

            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            return ComparisonResult(
                similarity_score=float(score),
                passed=score >= threshold,
                threshold=threshold,
                algorithm=ComparisonAlgorithm.SSIM,
                execution_time_ms=execution_time_ms,
                diff_regions=diff_regions,
                diff_mask=diff_mask,
            )

        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            return ComparisonResult(
                similarity_score=0.0,
                passed=False,
                threshold=threshold,
                algorithm=ComparisonAlgorithm.SSIM,
                execution_time_ms=execution_time_ms,
                error=str(e),
            )

    def compare_pixel_diff(
        self,
        baseline: np.ndarray,
        screenshot: np.ndarray,
        threshold: float = 0.99,
        color_tolerance: int = 5,
    ) -> ComparisonResult:
        """Compare images using pixel-by-pixel difference.

        This method compares each pixel and counts differences exceeding the
        color tolerance. Best for exact match verification where minor
        rendering differences should be ignored.

        Args:
            baseline: Baseline image (BGR)
            screenshot: Screenshot to compare (BGR)
            threshold: Minimum similarity score to pass (default 0.99)
            color_tolerance: Maximum color difference per channel (0-255) to ignore

        Returns:
            ComparisonResult with pixel match percentage and diff regions
        """
        start_time = time.perf_counter()

        try:
            # Ensure same size
            baseline, screenshot = self._resize_to_match(baseline, screenshot)

            # Calculate absolute difference
            diff = cv2.absdiff(baseline, screenshot)

            # Apply color tolerance - pixels within tolerance are considered matching
            if color_tolerance > 0:
                # Create mask of pixels exceeding tolerance (any channel)
                diff_gray = np.max(diff, axis=2) if len(diff.shape) == 3 else diff
                diff_mask = (diff_gray > color_tolerance).astype(np.uint8) * 255
            else:
                # Any difference counts
                diff_gray = np.max(diff, axis=2) if len(diff.shape) == 3 else diff
                diff_mask = (diff_gray > 0).astype(np.uint8) * 255

            # Calculate similarity (percentage of matching pixels)
            total_pixels = baseline.shape[0] * baseline.shape[1]
            diff_pixels = np.count_nonzero(diff_mask)
            matching_pixels = total_pixels - diff_pixels
            similarity_score = matching_pixels / total_pixels if total_pixels > 0 else 0.0

            # Extract diff regions
            diff_regions = self._extract_diff_regions(diff_mask)

            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            return ComparisonResult(
                similarity_score=similarity_score,
                passed=similarity_score >= threshold,
                threshold=threshold,
                algorithm=ComparisonAlgorithm.PIXEL_DIFF,
                execution_time_ms=execution_time_ms,
                diff_regions=diff_regions,
                diff_mask=diff_mask,
            )

        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            return ComparisonResult(
                similarity_score=0.0,
                passed=False,
                threshold=threshold,
                algorithm=ComparisonAlgorithm.PIXEL_DIFF,
                execution_time_ms=execution_time_ms,
                error=str(e),
            )

    def compare_perceptual_hash(
        self,
        baseline: np.ndarray,
        screenshot: np.ndarray,
        threshold: float = 0.90,
        hash_size: int = 16,
    ) -> ComparisonResult:
        """Compare images using perceptual hashing.

        Perceptual hashing creates a compact fingerprint of the image that's
        robust to minor changes. Best for detecting if content is fundamentally
        the same despite minor rendering differences.

        Args:
            baseline: Baseline image
            screenshot: Screenshot to compare
            threshold: Minimum similarity score to pass (default 0.90)
            hash_size: Size of the hash (larger = more precision, default 16)

        Returns:
            ComparisonResult with hash similarity score
        """
        start_time = time.perf_counter()

        if not IMAGEHASH_AVAILABLE:
            return ComparisonResult(
                similarity_score=0.0,
                passed=False,
                threshold=threshold,
                algorithm=ComparisonAlgorithm.PERCEPTUAL_HASH,
                execution_time_ms=0,
                error="imagehash not installed. Install with: pip install imagehash",
            )

        try:
            # Convert numpy arrays to PIL Images
            baseline_pil = self._numpy_to_pil(baseline)
            screenshot_pil = self._numpy_to_pil(screenshot)

            # Compute perceptual hashes
            baseline_hash = imagehash.phash(baseline_pil, hash_size=hash_size)
            screenshot_hash = imagehash.phash(screenshot_pil, hash_size=hash_size)

            # Calculate similarity (hash difference is hamming distance)
            max_distance = hash_size * hash_size  # Maximum possible hamming distance
            hash_distance = baseline_hash - screenshot_hash
            similarity_score = 1.0 - (hash_distance / max_distance)

            # For perceptual hash, we can't generate a pixel-level diff mask
            # but we can estimate regions by splitting into quadrants
            diff_regions = []
            if similarity_score < threshold:
                # Generate rough diff regions by comparing quadrant hashes
                diff_regions = self._estimate_diff_regions_from_quadrants(
                    baseline, screenshot, hash_size=8
                )

            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            return ComparisonResult(
                similarity_score=similarity_score,
                passed=similarity_score >= threshold,
                threshold=threshold,
                algorithm=ComparisonAlgorithm.PERCEPTUAL_HASH,
                execution_time_ms=execution_time_ms,
                diff_regions=diff_regions,
                diff_mask=None,  # Not available for perceptual hash
            )

        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            return ComparisonResult(
                similarity_score=0.0,
                passed=False,
                threshold=threshold,
                algorithm=ComparisonAlgorithm.PERCEPTUAL_HASH,
                execution_time_ms=execution_time_ms,
                error=str(e),
            )

    def apply_ignore_mask(
        self,
        image: np.ndarray,
        regions: list[IgnoreRegion] | list[dict],
    ) -> np.ndarray:
        """Apply ignore mask to image, setting specified regions to neutral color.

        Args:
            image: Input image (BGR or grayscale)
            regions: List of regions to ignore

        Returns:
            Image with ignore regions masked (filled with gray)
        """
        masked = image.copy()
        fill_color = (128, 128, 128) if len(image.shape) == 3 else 128

        for region in regions:
            if isinstance(region, dict):
                region = IgnoreRegion.from_dict(region)

            x1 = max(0, region.x)
            y1 = max(0, region.y)
            x2 = min(masked.shape[1], region.x + region.width)
            y2 = min(masked.shape[0], region.y + region.height)

            if x2 > x1 and y2 > y1:
                masked[y1:y2, x1:x2] = fill_color

        return masked

    def generate_diff_image(
        self,
        baseline: np.ndarray,
        screenshot: np.ndarray,
        diff_mask: np.ndarray | None = None,
        highlight_color: tuple[int, int, int] = (0, 0, 255),  # Red in BGR
        overlay_alpha: float = 0.5,
    ) -> bytes:
        """Generate a diff visualization image with changes highlighted.

        Creates an image showing the screenshot with changed regions highlighted
        in the specified color.

        Args:
            baseline: Baseline image
            screenshot: Current screenshot
            diff_mask: Binary mask of differences (if None, will be computed)
            highlight_color: BGR color for highlighting differences
            overlay_alpha: Transparency of the highlight overlay (0.0-1.0)

        Returns:
            PNG image bytes
        """
        # Ensure same size
        baseline, screenshot = self._resize_to_match(baseline, screenshot)

        # Compute diff mask if not provided
        if diff_mask is None:
            result = self.compare_pixel_diff(baseline, screenshot)
            diff_mask = result.diff_mask

        # Ensure screenshot is BGR for visualization
        if len(screenshot.shape) == 2:
            output = cv2.cvtColor(screenshot, cv2.COLOR_GRAY2BGR)
        else:
            output = screenshot.copy()

        # Create highlight overlay
        if diff_mask is not None and np.any(diff_mask):
            # Ensure diff_mask matches output dimensions
            if diff_mask.shape[:2] != output.shape[:2]:
                diff_mask = cv2.resize(
                    diff_mask, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_NEAREST
                )

            # Create colored overlay
            overlay = output.copy()
            overlay[diff_mask > 0] = highlight_color

            # Blend with original
            cv2.addWeighted(overlay, overlay_alpha, output, 1 - overlay_alpha, 0, output)

            # Draw contours around diff regions for clarity
            contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, highlight_color, 2)

        # Convert to PNG bytes
        success, png_data = cv2.imencode(".png", output)
        if success:
            return png_data.tobytes()
        else:
            raise RuntimeError("Failed to encode diff image as PNG")

    def compute_perceptual_hash(self, image: np.ndarray, hash_size: int = 16) -> str:
        """Compute perceptual hash for an image.

        The hash can be stored with baselines for quick pre-comparison
        filtering before running full comparison.

        Args:
            image: Input image
            hash_size: Size of the hash (default 16)

        Returns:
            Hex string representation of the perceptual hash
        """
        if not IMAGEHASH_AVAILABLE:
            raise RuntimeError("imagehash not installed")

        pil_image = self._numpy_to_pil(image)
        phash = imagehash.phash(pil_image, hash_size=hash_size)
        return str(phash)

    def hash_distance(self, hash1: str, hash2: str) -> int:
        """Calculate hamming distance between two perceptual hashes.

        Args:
            hash1: First hash string
            hash2: Second hash string

        Returns:
            Hamming distance (number of differing bits)
        """
        if not IMAGEHASH_AVAILABLE:
            raise RuntimeError("imagehash not installed")

        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return h1 - h2

    # Private helper methods

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 2:
            return image
        elif image.shape[2] == 4:
            # BGRA to grayscale
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            # BGR to grayscale
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _resize_to_match(self, img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Resize images to match dimensions (uses larger dimensions)."""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if h1 == h2 and w1 == w2:
            return img1, img2

        # Use the larger dimensions
        target_h = max(h1, h2)
        target_w = max(w1, w2)

        if h1 != target_h or w1 != target_w:
            img1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if h2 != target_h or w2 != target_w:
            img2 = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return img1, img2

    def _numpy_to_pil(self, image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image."""
        if len(image.shape) == 2:
            # Grayscale
            return Image.fromarray(image, mode="L")
        elif image.shape[2] == 3:
            # BGR to RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb, mode="RGB")
        elif image.shape[2] == 4:
            # BGRA to RGBA
            rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(rgba, mode="RGBA")
        else:
            raise ValueError(f"Unsupported image format: {image.shape}")

    def _extract_diff_regions(self, diff_mask: np.ndarray, min_area: int = 100) -> list[DiffRegion]:
        """Extract bounding box regions from diff mask.

        Args:
            diff_mask: Binary mask where white (255) indicates differences
            min_area: Minimum area in pixels to report as a region

        Returns:
            List of DiffRegion objects
        """
        if diff_mask is None or not np.any(diff_mask):
            return []

        # Find contours of diff areas
        contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area >= min_area:
                # Count actual diff pixels in this region
                region_mask = diff_mask[y : y + h, x : x + w]
                pixel_count = int(np.count_nonzero(region_mask))
                change_percentage = pixel_count / area if area > 0 else 0.0

                regions.append(
                    DiffRegion(
                        x=int(x),
                        y=int(y),
                        width=int(w),
                        height=int(h),
                        change_percentage=change_percentage,
                        pixel_count=pixel_count,
                    )
                )

        # Sort by area (largest first)
        regions.sort(key=lambda r: r.width * r.height, reverse=True)
        return regions

    def _estimate_diff_regions_from_quadrants(
        self, baseline: np.ndarray, screenshot: np.ndarray, hash_size: int = 8
    ) -> list[DiffRegion]:
        """Estimate diff regions by comparing image quadrants.

        Used for perceptual hash comparison where pixel-level diff isn't available.
        """
        if not IMAGEHASH_AVAILABLE:
            return []

        regions = []
        h, w = baseline.shape[:2]

        # Split into 4 quadrants
        quadrants = [
            (0, 0, w // 2, h // 2),  # Top-left
            (w // 2, 0, w - w // 2, h // 2),  # Top-right
            (0, h // 2, w // 2, h - h // 2),  # Bottom-left
            (w // 2, h // 2, w - w // 2, h - h // 2),  # Bottom-right
        ]

        for x, y, qw, qh in quadrants:
            if qw <= 0 or qh <= 0:
                continue

            baseline_quad = baseline[y : y + qh, x : x + qw]
            screenshot_quad = screenshot[y : y + qh, x : x + qw]

            # Compare quadrant hashes
            b_pil = self._numpy_to_pil(baseline_quad)
            s_pil = self._numpy_to_pil(screenshot_quad)

            b_hash = imagehash.phash(b_pil, hash_size=hash_size)
            s_hash = imagehash.phash(s_pil, hash_size=hash_size)

            distance = b_hash - s_hash
            max_dist = hash_size * hash_size
            similarity = 1.0 - (distance / max_dist)

            # If quadrant differs significantly, report it
            if similarity < 0.9:
                regions.append(
                    DiffRegion(
                        x=x,
                        y=y,
                        width=qw,
                        height=qh,
                        change_percentage=1.0 - similarity,
                        pixel_count=int((1.0 - similarity) * qw * qh),
                    )
                )

        return regions


# Convenience function for one-off comparisons
def compare_images(
    baseline: np.ndarray,
    screenshot: np.ndarray,
    algorithm: ComparisonAlgorithm | str = ComparisonAlgorithm.SSIM,
    threshold: float | None = None,
    ignore_regions: list[IgnoreRegion] | list[dict] | None = None,
) -> ComparisonResult:
    """Compare two images using the specified algorithm.

    Convenience function that creates a VisualComparator instance.

    Args:
        baseline: Baseline image
        screenshot: Screenshot to compare
        algorithm: Comparison algorithm (default: SSIM)
        threshold: Similarity threshold (uses algorithm default if not specified)
        ignore_regions: Regions to mask out before comparison

    Returns:
        ComparisonResult with similarity score and diff information
    """
    comparator = VisualComparator()
    return comparator.compare(
        baseline=baseline,
        screenshot=screenshot,
        algorithm=algorithm,
        threshold=threshold,
        ignore_regions=ignore_regions,
    )
