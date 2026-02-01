"""Application profile tuning for click-to-template system.

This module provides automatic tuning of detection parameters
based on sample screenshots from a specific application.
"""

import logging
import time
from datetime import datetime

import cv2
import numpy as np

from .application_profile import ApplicationProfile, TuningMetrics, TuningResult
from .boundary_finder import ElementBoundaryFinder
from .models import DetectionStrategy, InferenceConfig, InferredBoundingBox

logger = logging.getLogger(__name__)


class ApplicationTuner:
    """Learns optimal detection parameters for applications.

    Analyzes sample screenshots to determine the best detection
    parameters for a specific application's UI style.

    Example:
        >>> tuner = ApplicationTuner()
        >>> result = tuner.tune_from_samples(
        ...     screenshots=[img1, img2, img3],
        ...     known_elements=ground_truth_boxes,
        ... )
        >>> if result.success:
        ...     profile = ApplicationProfile(
        ...         id="my-app",
        ...         name="My Application",
        ...         inference_config=result.config,
        ...     )
    """

    # Edge threshold presets to test
    EDGE_THRESHOLD_PRESETS = [
        (30, 100),
        (50, 150),
        (70, 200),
        (100, 250),
        (40, 120),
        (60, 180),
    ]

    # Color tolerance values to test
    COLOR_TOLERANCE_VALUES = [15, 20, 25, 30, 40, 50]

    def __init__(self, base_config: InferenceConfig | None = None) -> None:
        """Initialize the tuner.

        Args:
            base_config: Base configuration to start from.
        """
        self.base_config = base_config or InferenceConfig()

    def tune_from_samples(
        self,
        screenshots: list[np.ndarray],
        known_elements: list[InferredBoundingBox] | None = None,
        click_locations: list[tuple[int, int]] | None = None,
    ) -> TuningResult:
        """Tune parameters from sample screenshots.

        Args:
            screenshots: List of sample screenshots (BGR format).
            known_elements: Optional ground truth bounding boxes.
            click_locations: Optional click locations to test detection on.

        Returns:
            TuningResult with optimized configuration and metrics.
        """
        if not screenshots:
            return TuningResult(
                config=self.base_config,
                success=False,
                error_message="No screenshots provided",
            )

        logger.info(f"Starting tuning with {len(screenshots)} samples")
        start_time = time.time()

        try:
            # Step 1: Tune edge thresholds
            edge_thresholds = self.tune_edge_thresholds(screenshots)
            logger.info(f"Optimal edge thresholds: {edge_thresholds}")

            # Step 2: Analyze color characteristics
            color_tolerance = self.tune_color_tolerance(screenshots, click_locations)
            logger.info(f"Optimal color tolerance: {color_tolerance}")

            # Step 3: Rank detection strategies
            strategy_rankings = self.rank_strategies(screenshots, known_elements, click_locations)
            logger.info(f"Strategy rankings: {strategy_rankings[:3]}")

            # Step 4: Analyze element characteristics
            avg_size, color_ranges = self.analyze_element_characteristics(
                screenshots, known_elements
            )
            logger.info(f"Average element size: {avg_size}")

            # Build optimized config
            config = InferenceConfig(
                search_radius=self.base_config.search_radius,
                min_element_size=self.base_config.min_element_size,
                max_element_size=self.base_config.max_element_size,
                edge_threshold_low=edge_thresholds[0],
                edge_threshold_high=edge_thresholds[1],
                color_tolerance=color_tolerance,
                contour_area_min=self.base_config.contour_area_min,
                fallback_box_size=self.base_config.fallback_box_size,
                use_fallback=self.base_config.use_fallback,
                preferred_strategies=[s for s, _ in strategy_rankings[:4]],
                enable_mask_generation=self.base_config.enable_mask_generation,
                enable_element_classification=self.base_config.enable_element_classification,
                merge_nearby_boundaries=self.base_config.merge_nearby_boundaries,
                merge_gap=self.base_config.merge_gap,
            )

            # Build metrics
            metrics = TuningMetrics(
                sample_count=len(screenshots),
                edge_score=self._get_strategy_score(
                    strategy_rankings, DetectionStrategy.EDGE_BASED
                ),
                contour_score=self._get_strategy_score(
                    strategy_rankings, DetectionStrategy.CONTOUR_BASED
                ),
                color_score=self._get_strategy_score(
                    strategy_rankings, DetectionStrategy.COLOR_SEGMENTATION
                ),
                flood_fill_score=self._get_strategy_score(
                    strategy_rankings, DetectionStrategy.FLOOD_FILL
                ),
                gradient_score=self._get_strategy_score(
                    strategy_rankings, DetectionStrategy.GRADIENT_BASED
                ),
                tuning_iterations=1,
                last_tuned_at=datetime.now(),
            )

            elapsed = time.time() - start_time
            logger.info(f"Tuning completed in {elapsed:.2f}s")

            return TuningResult(
                config=config,
                strategy_rankings=strategy_rankings,
                metrics=metrics,
                success=True,
            )

        except Exception as e:
            logger.error(f"Tuning failed: {e}")
            return TuningResult(
                config=self.base_config,
                success=False,
                error_message=str(e),
            )

    def tune_edge_thresholds(
        self,
        screenshots: list[np.ndarray],
    ) -> tuple[int, int]:
        """Find optimal Canny edge detection thresholds.

        Tests multiple threshold presets and scores them based on:
        - Edge continuity (prefer closed contours)
        - Number of contours (not too few, not too many)
        - Noise level (less noise is better)

        Args:
            screenshots: Sample screenshots.

        Returns:
            Tuple of (low_threshold, high_threshold).
        """
        best_score = -1.0
        best_thresholds = (50, 150)

        for low, high in self.EDGE_THRESHOLD_PRESETS:
            total_score = 0.0

            for screenshot in screenshots:
                if len(screenshot.shape) == 3:
                    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
                else:
                    gray = screenshot

                edges = cv2.Canny(gray, low, high)

                # Score based on contour characteristics
                contours, hierarchy = cv2.findContours(
                    edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )

                # Prefer moderate number of contours
                contour_count = len(contours)
                count_score = 1.0 - abs(contour_count - 50) / 100
                count_score = max(0, min(1, count_score))

                # Check edge density (prefer 5-15% edge pixels)
                edge_density = np.sum(edges > 0) / edges.size
                density_score = 1.0 - abs(edge_density - 0.10) / 0.10
                density_score = max(0, min(1, density_score))

                # Check for closed contours
                closed_count = sum(
                    1 for c in contours if cv2.arcLength(c, True) > 0 and cv2.contourArea(c) > 100
                )
                closed_ratio = closed_count / max(1, contour_count)

                score = 0.3 * count_score + 0.3 * density_score + 0.4 * closed_ratio
                total_score += score

            avg_score = total_score / len(screenshots)

            if avg_score > best_score:
                best_score = avg_score
                best_thresholds = (low, high)

        return best_thresholds

    def tune_color_tolerance(
        self,
        screenshots: list[np.ndarray],
        click_locations: list[tuple[int, int]] | None = None,
    ) -> int:
        """Find optimal color tolerance for segmentation.

        Analyzes intra-region color variance to determine appropriate
        tolerance that captures 95% of region pixels.

        Args:
            screenshots: Sample screenshots.
            click_locations: Optional click locations to sample colors from.

        Returns:
            Optimal color tolerance value.
        """
        if not click_locations:
            # Sample random points as pseudo-clicks
            click_locations = []
            for screenshot in screenshots:
                h, w = screenshot.shape[:2]
                # Sample 5 random interior points
                for _ in range(5):
                    x = np.random.randint(w // 4, 3 * w // 4)
                    y = np.random.randint(h // 4, 3 * h // 4)
                    click_locations.append((x, y))

        # Calculate color variance at click regions
        variances = []

        for i, (x, y) in enumerate(click_locations):
            screenshot_idx = i % len(screenshots)
            screenshot = screenshots[screenshot_idx]

            h, w = screenshot.shape[:2]
            if not (0 <= x < w and 0 <= y < h):
                continue

            # Sample a small region around the click
            radius = 25
            x1 = max(0, x - radius)
            y1 = max(0, y - radius)
            x2 = min(w, x + radius)
            y2 = min(h, y + radius)

            region = screenshot[y1:y2, x1:x2]

            if region.size == 0:
                continue

            if len(region.shape) == 3:
                # Calculate per-channel variance
                for c in range(3):
                    variances.append(np.std(region[:, :, c]))
            else:
                variances.append(np.std(region))

        if not variances:
            return 30  # Default

        # Use 95th percentile of variance as tolerance
        tolerance = int(np.percentile(variances, 95))

        # Clamp to reasonable range
        return max(15, min(60, tolerance))

    def rank_strategies(
        self,
        screenshots: list[np.ndarray],
        known_elements: list[InferredBoundingBox] | None = None,
        click_locations: list[tuple[int, int]] | None = None,
    ) -> list[tuple[DetectionStrategy, float]]:
        """Rank detection strategies by effectiveness.

        Args:
            screenshots: Sample screenshots.
            known_elements: Optional ground truth bounding boxes.
            click_locations: Optional click locations to test.

        Returns:
            List of (strategy, score) tuples, sorted by score descending.
        """
        strategies = [
            DetectionStrategy.CONTOUR_BASED,
            DetectionStrategy.EDGE_BASED,
            DetectionStrategy.COLOR_SEGMENTATION,
            DetectionStrategy.FLOOD_FILL,
            DetectionStrategy.GRADIENT_BASED,
        ]

        scores: dict[DetectionStrategy, list[float]] = {s: [] for s in strategies}

        # Generate test clicks if not provided
        if not click_locations:
            click_locations = []
            for screenshot in screenshots:
                h, w = screenshot.shape[:2]
                for _ in range(3):
                    x = np.random.randint(w // 4, 3 * w // 4)
                    y = np.random.randint(h // 4, 3 * h // 4)
                    click_locations.append((x, y))

        # Test each strategy
        for strategy in strategies:
            config = InferenceConfig(
                preferred_strategies=[strategy],
                use_fallback=False,
            )
            finder = ElementBoundaryFinder(config)

            for i, (x, y) in enumerate(click_locations):
                screenshot_idx = i % len(screenshots)
                screenshot = screenshots[screenshot_idx]

                h, w = screenshot.shape[:2]
                if not (0 <= x < w and 0 <= y < h):
                    continue

                try:
                    candidates = finder.find_boundaries(screenshot, (x, y), [strategy])

                    if candidates:
                        # Score based on confidence and rectangularity
                        best = candidates[0]
                        score = best.confidence

                        # Bonus for reasonable size
                        area = best.area
                        if 500 < area < 50000:
                            score *= 1.2

                        # If we have ground truth, calculate IoU
                        if known_elements:
                            best_iou = self._best_iou(best, known_elements)
                            score = (score + best_iou) / 2

                        scores[strategy].append(min(1.0, score))
                    else:
                        scores[strategy].append(0.0)

                except Exception:
                    scores[strategy].append(0.0)

        # Calculate average scores
        avg_scores = []
        for strategy in strategies:
            if scores[strategy]:
                avg = sum(scores[strategy]) / len(scores[strategy])
            else:
                avg = 0.0
            avg_scores.append((strategy, avg))

        # Sort by score descending
        avg_scores.sort(key=lambda x: -x[1])

        return avg_scores

    def analyze_element_characteristics(
        self,
        screenshots: list[np.ndarray],
        known_elements: list[InferredBoundingBox] | None = None,
    ) -> tuple[tuple[int, int], list[tuple[tuple[int, int, int], tuple[int, int, int]]]]:
        """Analyze common element characteristics in screenshots.

        Args:
            screenshots: Sample screenshots.
            known_elements: Optional known bounding boxes.

        Returns:
            Tuple of (average_size, common_color_ranges).
        """
        widths = []
        heights = []
        colors = []

        if known_elements:
            for elem in known_elements:
                widths.append(elem.width)
                heights.append(elem.height)

        # Also detect elements from screenshots
        finder = ElementBoundaryFinder(self.base_config)

        for screenshot in screenshots:
            h, w = screenshot.shape[:2]

            # Sample grid of points
            for gx in range(5):
                for gy in range(5):
                    x = int((gx + 0.5) * w / 5)
                    y = int((gy + 0.5) * h / 5)

                    candidates = finder.find_boundaries(screenshot, (x, y))

                    for c in candidates[:1]:  # Just the best
                        widths.append(c.width)
                        heights.append(c.height)

                        # Sample color at center
                        if c.pixel_data is not None and c.pixel_data.size > 0:
                            center_color = c.pixel_data[c.height // 2, c.width // 2]
                            if len(center_color.shape) == 0:
                                center_color = [center_color] * 3
                            colors.append(tuple(center_color))

        # Calculate average size
        if widths and heights:
            avg_width = int(np.median(widths))
            avg_height = int(np.median(heights))
        else:
            avg_width, avg_height = 60, 30

        # Find common color ranges
        color_ranges = self._cluster_colors(colors)

        return (avg_width, avg_height), color_ranges

    def _cluster_colors(
        self,
        colors: list[tuple[int, ...]],
        n_clusters: int = 5,
    ) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
        """Cluster colors and return range bounds."""
        if not colors or len(colors) < n_clusters:
            return []

        # Convert to numpy array
        color_array = np.array(colors, dtype=np.float32)

        # Convert to HSV if RGB/BGR
        if color_array.shape[1] == 3:
            # Reshape for OpenCV
            color_array_reshaped = color_array.reshape(-1, 1, 3).astype(np.uint8)
            hsv = cv2.cvtColor(color_array_reshaped, cv2.COLOR_BGR2HSV)
            color_array = hsv.reshape(-1, 3).astype(np.float32)

        # Simple K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            color_array, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Build ranges around centers
        ranges = []
        for i, _center in enumerate(centers):
            cluster_colors = color_array[labels.flatten() == i]
            if len(cluster_colors) == 0:
                continue

            low = tuple(int(max(0, np.percentile(cluster_colors[:, j], 5))) for j in range(3))
            high = tuple(int(min(255, np.percentile(cluster_colors[:, j], 95))) for j in range(3))
            ranges.append((low, high))

        return ranges

    def _get_strategy_score(
        self,
        rankings: list[tuple[DetectionStrategy, float]],
        strategy: DetectionStrategy,
    ) -> float:
        """Get score for a specific strategy from rankings."""
        for s, score in rankings:
            if s == strategy:
                return score
        return 0.0

    def _best_iou(
        self,
        candidate: InferredBoundingBox,
        ground_truth: list[InferredBoundingBox],
    ) -> float:
        """Calculate best IoU with any ground truth box."""
        best = 0.0

        for gt in ground_truth:
            # Calculate intersection
            x1 = max(candidate.x, gt.x)
            y1 = max(candidate.y, gt.y)
            x2 = min(candidate.x2, gt.x2)
            y2 = min(candidate.y2, gt.y2)

            if x2 <= x1 or y2 <= y1:
                continue

            intersection = (x2 - x1) * (y2 - y1)
            union = candidate.area + gt.area - intersection

            iou = intersection / union if union > 0 else 0
            best = max(best, iou)

        return best

    def tune_incrementally(
        self,
        profile: ApplicationProfile,
        new_screenshots: list[np.ndarray],
        success_feedback: list[bool] | None = None,
    ) -> TuningResult:
        """Incrementally tune an existing profile with new samples.

        Args:
            profile: Existing profile to refine.
            new_screenshots: New sample screenshots.
            success_feedback: Optional feedback on which detections succeeded.

        Returns:
            TuningResult with updated configuration.
        """
        # Start from existing config
        self.base_config = profile.inference_config

        # Tune with new samples
        result = self.tune_from_samples(new_screenshots)

        if result.success:
            # Blend old and new metrics
            old_weight = min(0.7, profile.tuning_metrics.sample_count / 100)
            new_weight = 1 - old_weight

            result.metrics.sample_count = profile.tuning_metrics.sample_count + len(new_screenshots)
            result.metrics.tuning_iterations = profile.tuning_metrics.tuning_iterations + 1

            # Weighted average of scores
            for attr in ["edge_score", "contour_score", "color_score", "flood_fill_score"]:
                old_val = getattr(profile.tuning_metrics, attr)
                new_val = getattr(result.metrics, attr)
                setattr(result.metrics, attr, old_weight * old_val + new_weight * new_val)

        return result
