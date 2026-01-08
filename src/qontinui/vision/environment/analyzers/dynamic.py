"""Dynamic Region Detector for GUI Environment Discovery.

Detects regions that change dynamically including:
- Always-changing regions (timestamps, counters)
- Conditionally-changing regions (content areas)
- Animation regions (spinners, progress bars)

Uses frame-over-frame comparison and motion analysis.
"""

import logging
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.environment import (
    AnimationRegion,
    AnimationType,
    BoundingBox,
    ChangeFrequency,
    DynamicRegion,
    DynamicRegions,
)

from qontinui.vision.environment.analyzers.base import BaseAnalyzer

logger = logging.getLogger(__name__)

# Patterns that suggest dynamic content
DYNAMIC_PATTERNS = {
    "timestamp": r"\d{1,2}:\d{2}(:\d{2})?(\s*[AP]M)?",
    "date": r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}",
    "counter": r"^\d+$",
    "percentage": r"\d+(\.\d+)?%",
    "elapsed": r"\d+[smh]",
}


class DynamicRegionDetector(BaseAnalyzer[DynamicRegions]):
    """Detects dynamic regions through frame comparison and motion analysis.

    Analyzes idle frames to find always-changing regions and
    before/after action pairs to find conditionally-changing regions.
    """

    def __init__(
        self,
        variance_threshold: float = 0.1,
        min_region_size: int = 20,
        animation_fps: int = 10,
    ) -> None:
        """Initialize the dynamic region detector.

        Args:
            variance_threshold: Pixel variance threshold for change detection.
            min_region_size: Minimum region size to consider (pixels).
            animation_fps: Expected FPS for animation detection.
        """
        super().__init__("DynamicRegionDetector")
        self.variance_threshold = variance_threshold
        self.min_region_size = min_region_size
        self.animation_fps = animation_fps

    async def analyze(
        self,
        screenshots: list[NDArray[np.uint8]],
        idle_frames: list[NDArray[np.uint8]] | None = None,
        action_pairs: list[tuple[NDArray[np.uint8], NDArray[np.uint8]]] | None = None,
        ocr_results: list[list[dict[str, Any]]] | None = None,
        **kwargs: Any,
    ) -> DynamicRegions:
        """Analyze screenshots to detect dynamic regions.

        Args:
            screenshots: List of screenshots for basic analysis.
            idle_frames: Optional frames captured during idle state (for animation detection).
            action_pairs: Optional (before, after) screenshot pairs for conditional changes.
            ocr_results: Optional OCR results for pattern matching.

        Returns:
            DynamicRegions with detected regions.
        """
        self.reset()

        always_changing: list[DynamicRegion] = []
        conditionally_changing: list[DynamicRegion] = []
        animation_regions: list[AnimationRegion] = []

        # Detect always-changing regions from idle frames or consecutive screenshots
        frames_to_analyze = idle_frames if idle_frames else screenshots
        if len(frames_to_analyze) >= 2:
            always_changing = self._detect_always_changing(frames_to_analyze, ocr_results)

        # Detect conditionally-changing regions from action pairs
        if action_pairs:
            conditionally_changing = self._detect_conditional_changes(action_pairs)

        # Detect animations from idle frames
        if idle_frames and len(idle_frames) >= 5:
            animation_regions = self._detect_animations(idle_frames)

        self._screenshots_analyzed = len(screenshots)
        idle_count = len(idle_frames) if idle_frames else 0
        action_count = len(action_pairs) if action_pairs else 0

        self.confidence = self._calculate_confidence(
            len(screenshots) + idle_count + action_count,
            min_samples=3,
            optimal_samples=20,
        )

        return DynamicRegions(
            always_changing=always_changing,
            conditionally_changing=conditionally_changing,
            animation_regions=animation_regions,
            idle_frames_analyzed=idle_count,
            action_pairs_analyzed=action_count,
            confidence=self.confidence,
        )

    def _detect_always_changing(
        self,
        frames: list[NDArray[np.uint8]],
        ocr_results: list[list[dict[str, Any]]] | None = None,
    ) -> list[DynamicRegion]:
        """Detect regions that always change between frames.

        Args:
            frames: List of frames to compare.
            ocr_results: Optional OCR results for pattern detection.

        Returns:
            List of always-changing DynamicRegion objects.
        """
        regions = []

        if len(frames) < 2:
            return regions

        # Ensure all frames are BGR
        frames = [self._ensure_bgr(f) for f in frames]

        # Compute per-pixel variance across frames
        variance_map = self._compute_variance_map(frames)

        # Find high-variance regions
        high_variance_regions = self._find_high_variance_regions(variance_map)

        # Classify regions
        for bbox in high_variance_regions:
            x, y, w, h = bbox

            # Try to match patterns using OCR
            pattern = None
            regex_pattern = None

            if ocr_results and len(ocr_results) > 0:
                pattern, regex_pattern = self._detect_pattern(bbox, frames[0], ocr_results[0])

            # Determine if this should be auto-masked
            auto_mask = pattern in ["timestamp", "counter", "elapsed", "percentage"]

            regions.append(
                DynamicRegion(
                    bounds=BoundingBox(x=x, y=y, width=w, height=h),
                    change_frequency=ChangeFrequency.CONTINUOUS,
                    pattern=pattern,
                    regex_pattern=regex_pattern,
                    auto_mask=auto_mask,
                )
            )

        return regions

    def _compute_variance_map(
        self,
        frames: list[NDArray[np.uint8]],
    ) -> NDArray[np.float64]:
        """Compute per-pixel variance across frames.

        Args:
            frames: List of BGR frames (same size).

        Returns:
            2D array of variance values.
        """
        # Stack frames
        stacked = np.stack(frames, axis=0).astype(np.float64)

        # Compute variance across frames for each pixel
        # Average across color channels
        variance = np.var(stacked, axis=0).mean(axis=2)

        return variance

    def _find_high_variance_regions(
        self,
        variance_map: NDArray[np.float64],
    ) -> list[tuple[int, int, int, int]]:
        """Find connected regions with high variance.

        Args:
            variance_map: Per-pixel variance values.

        Returns:
            List of bounding boxes (x, y, width, height).
        """
        regions = []

        # Threshold variance
        threshold = np.percentile(variance_map, 95)  # Top 5% variance
        threshold = max(threshold, self.variance_threshold * 255)

        binary = (variance_map > threshold).astype(np.uint8)

        try:
            import cv2

            # Dilate to connect nearby pixels
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w >= self.min_region_size and h >= self.min_region_size:
                    regions.append((x, y, w, h))

        except ImportError:
            # Simple fallback without cv2
            regions = self._simple_region_finding(binary)

        return regions

    def _simple_region_finding(
        self,
        binary: NDArray[np.uint8],
    ) -> list[tuple[int, int, int, int]]:
        """Simple region finding without OpenCV.

        Args:
            binary: Binary mask.

        Returns:
            List of bounding boxes.
        """
        # Find non-zero pixel bounds
        nonzero = np.where(binary > 0)
        if len(nonzero[0]) == 0:
            return []

        y_min, y_max = nonzero[0].min(), nonzero[0].max()
        x_min, x_max = nonzero[1].min(), nonzero[1].max()

        w = x_max - x_min
        h = y_max - y_min

        if w >= self.min_region_size and h >= self.min_region_size:
            return [(x_min, y_min, w, h)]
        return []

    def _detect_pattern(
        self,
        bbox: tuple[int, int, int, int],
        screenshot: NDArray[np.uint8],
        ocr_result: list[dict[str, Any]],
    ) -> tuple[str | None, str | None]:
        """Detect the pattern type of text in a region.

        Args:
            bbox: Region bounding box.
            screenshot: Screenshot for context.
            ocr_result: OCR results.

        Returns:
            Tuple of (pattern_name, regex_pattern) or (None, None).
        """
        x, y, w, h = bbox

        # Find text in this region
        for item in ocr_result:
            item_bbox = item.get("bbox")
            if not item_bbox:
                continue

            # Get item position
            if len(item_bbox) == 4 and not isinstance(item_bbox[0], (list, tuple)):
                ix, iy, iw, ih = item_bbox
            else:
                continue

            # Check if text is inside the dynamic region
            if x <= ix < x + w and y <= iy < y + h:
                text = item.get("text", "")
                if text:
                    # Try to match patterns
                    for pattern_name, regex in DYNAMIC_PATTERNS.items():
                        if re.match(regex, text.strip()):
                            return pattern_name, regex

        return None, None

    def _detect_conditional_changes(
        self,
        action_pairs: list[tuple[NDArray[np.uint8], NDArray[np.uint8]]],
    ) -> list[DynamicRegion]:
        """Detect regions that change conditionally (after actions).

        Args:
            action_pairs: List of (before, after) screenshot pairs.

        Returns:
            List of conditionally-changing DynamicRegion objects.
        """
        regions = []
        all_change_regions: list[tuple[int, int, int, int]] = []

        for before, after in action_pairs:
            before = self._ensure_bgr(before)
            after = self._ensure_bgr(after)

            # Compute difference
            diff = np.abs(before.astype(np.int16) - after.astype(np.int16))
            diff_gray = diff.mean(axis=2)

            # Find changed regions
            change_regions = self._find_high_variance_regions(diff_gray)
            all_change_regions.extend(change_regions)

        # Deduplicate and merge overlapping regions
        merged = self._merge_overlapping_regions(all_change_regions)

        for x, y, w, h in merged:
            regions.append(
                DynamicRegion(
                    bounds=BoundingBox(x=x, y=y, width=w, height=h),
                    change_frequency=ChangeFrequency.ON_ACTION,
                    pattern="content_area",
                    auto_mask=False,  # Don't auto-mask action-dependent content
                )
            )

        return regions

    def _merge_overlapping_regions(
        self,
        regions: list[tuple[int, int, int, int]],
        overlap_threshold: float = 0.5,
    ) -> list[tuple[int, int, int, int]]:
        """Merge overlapping regions.

        Args:
            regions: List of bounding boxes.
            overlap_threshold: IoU threshold for merging.

        Returns:
            List of merged bounding boxes.
        """
        if not regions:
            return []

        # Simple greedy merging
        merged = []
        used = set()

        for i, r1 in enumerate(regions):
            if i in used:
                continue

            x1, y1, w1, h1 = r1
            merged_box = [x1, y1, x1 + w1, y1 + h1]

            for j, r2 in enumerate(regions[i + 1 :], i + 1):
                if j in used:
                    continue

                x2, y2, w2, h2 = r2

                # Check overlap
                if self._boxes_overlap(r1, r2):
                    # Merge
                    merged_box[0] = min(merged_box[0], x2)
                    merged_box[1] = min(merged_box[1], y2)
                    merged_box[2] = max(merged_box[2], x2 + w2)
                    merged_box[3] = max(merged_box[3], y2 + h2)
                    used.add(j)

            merged.append(
                (
                    merged_box[0],
                    merged_box[1],
                    merged_box[2] - merged_box[0],
                    merged_box[3] - merged_box[1],
                )
            )

        return merged

    def _boxes_overlap(
        self,
        box1: tuple[int, int, int, int],
        box2: tuple[int, int, int, int],
    ) -> bool:
        """Check if two boxes overlap.

        Args:
            box1: First box (x, y, width, height).
            box2: Second box (x, y, width, height).

        Returns:
            True if boxes overlap.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    def _detect_animations(
        self,
        idle_frames: list[NDArray[np.uint8]],
    ) -> list[AnimationRegion]:
        """Detect animation regions from idle frames.

        Args:
            idle_frames: Frames captured during idle state.

        Returns:
            List of AnimationRegion objects.
        """
        animations = []

        if len(idle_frames) < 5:
            return animations

        frames = [self._ensure_bgr(f) for f in idle_frames]

        # Find regions with periodic changes
        variance_map = self._compute_variance_map(frames)
        high_var_regions = self._find_high_variance_regions(variance_map)

        for bbox in high_var_regions:
            animation_type = self._classify_animation(frames, bbox)

            if animation_type != AnimationType.UNKNOWN:
                x, y, w, h = bbox
                animations.append(
                    AnimationRegion(
                        bounds=BoundingBox(x=x, y=y, width=w, height=h),
                        animation_type=animation_type,
                        is_continuous=True,
                    )
                )

        return animations

    def _classify_animation(
        self,
        frames: list[NDArray[np.uint8]],
        bbox: tuple[int, int, int, int],
    ) -> AnimationType:
        """Classify the type of animation in a region.

        Args:
            frames: List of frames.
            bbox: Animation region bounding box.

        Returns:
            AnimationType classification.
        """
        x, y, w, h = bbox

        # Extract regions from each frame
        regions = [f[y : y + h, x : x + w] for f in frames]

        # Check for rotation (spinner)
        if self._is_rotation_animation(regions):
            return AnimationType.SPINNER

        # Check for linear progress
        if self._is_progress_animation(regions):
            return AnimationType.PROGRESS

        # Check for pulsing (opacity changes)
        if self._is_pulse_animation(regions):
            return AnimationType.PULSE

        return AnimationType.UNKNOWN

    def _is_rotation_animation(
        self,
        regions: list[NDArray[np.uint8]],
    ) -> bool:
        """Check if regions show rotation animation.

        Args:
            regions: Cropped region from each frame.

        Returns:
            True if rotation detected.
        """
        # Simple heuristic: check if center of mass rotates
        try:
            import cv2

            centers = []
            for region in regions:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

                # Find center of mass
                moments = cv2.moments(binary)
                if moments["m00"] > 0:
                    cx = moments["m10"] / moments["m00"]
                    cy = moments["m01"] / moments["m00"]
                    centers.append((cx, cy))

            if len(centers) >= 3:
                # Check if centers trace a circular path
                # Simplified: check variance of distances from mean
                mean_x = sum(c[0] for c in centers) / len(centers)
                mean_y = sum(c[1] for c in centers) / len(centers)
                distances = [np.sqrt((c[0] - mean_x) ** 2 + (c[1] - mean_y) ** 2) for c in centers]
                if np.std(distances) < np.mean(distances) * 0.3:
                    return True

        except ImportError:
            pass

        return False

    def _is_progress_animation(
        self,
        regions: list[NDArray[np.uint8]],
    ) -> bool:
        """Check if regions show progress bar animation.

        Args:
            regions: Cropped region from each frame.

        Returns:
            True if progress animation detected.
        """
        # Check for horizontal growth pattern
        fill_ratios = []

        for region in regions:
            # Count non-background pixels
            gray = region.mean(axis=2) if len(region.shape) == 3 else region
            threshold = gray.mean()
            filled = (gray > threshold).sum() / gray.size
            fill_ratios.append(filled)

        # Progress bar: fill ratio should increase monotonically
        if len(fill_ratios) >= 3:
            diffs = [fill_ratios[i + 1] - fill_ratios[i] for i in range(len(fill_ratios) - 1)]
            if all(d >= 0 for d in diffs) and sum(diffs) > 0.1:
                return True

        return False

    def _is_pulse_animation(
        self,
        regions: list[NDArray[np.uint8]],
    ) -> bool:
        """Check if regions show pulsing animation.

        Args:
            regions: Cropped region from each frame.

        Returns:
            True if pulse animation detected.
        """
        # Check for periodic brightness changes
        brightnesses = [r.mean() for r in regions]

        if len(brightnesses) >= 4:
            # Look for oscillation
            diffs = [brightnesses[i + 1] - brightnesses[i] for i in range(len(brightnesses) - 1)]

            # Pulse: should have alternating positive/negative diffs
            sign_changes = sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0)

            if sign_changes >= len(diffs) * 0.5:
                return True

        return False
