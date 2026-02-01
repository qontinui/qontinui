"""Batch boundary detection with parallel processing.

This module provides efficient parallel processing for detecting
element boundaries across multiple frames and click locations.
"""

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .application_profile import ApplicationProfile
from .boundary_finder import ElementBoundaryFinder
from .context_analyzer import ClickContextAnalyzer
from .models import DetectionStrategy, InferenceConfig, InferenceResult, InferredBoundingBox

logger = logging.getLogger(__name__)


@dataclass
class BatchDetectionItem:
    """A single item in a batch detection request.

    Attributes:
        frame: The video frame or screenshot.
        click_location: (x, y) coordinates of the click.
        frame_id: Optional identifier for the frame.
        metadata: Additional metadata to preserve through processing.
    """

    frame: np.ndarray
    click_location: tuple[int, int]
    frame_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchDetectionResult:
    """Result of batch detection.

    Attributes:
        item: The original BatchDetectionItem.
        result: The InferenceResult from detection.
        processing_time_ms: Time taken to process this item.
        error: Error message if processing failed.
    """

    item: BatchDetectionItem
    result: InferenceResult | None
    processing_time_ms: float = 0.0
    error: str | None = None


class BatchDetector:
    """Parallel boundary detection for multiple frames.

    Processes multiple frames in parallel using a thread pool.
    Includes optional caching to avoid reprocessing identical regions.

    Example:
        >>> detector = BatchDetector(max_workers=4)
        >>> items = [
        ...     BatchDetectionItem(frame1, (100, 200)),
        ...     BatchDetectionItem(frame2, (150, 250)),
        ... ]
        >>> results = detector.detect_batch(items)
        >>> for r in results:
        ...     print(f"Frame {r.item.frame_id}: {r.result.primary_bbox}")
    """

    def __init__(
        self,
        max_workers: int = 4,
        config: InferenceConfig | None = None,
        profile: ApplicationProfile | None = None,
        enable_cache: bool = True,
        cache_max_size: int = 1000,
    ) -> None:
        """Initialize the batch detector.

        Args:
            max_workers: Maximum number of parallel workers.
            config: Inference configuration.
            profile: Application profile for tuned parameters.
            enable_cache: Whether to cache detection results.
            cache_max_size: Maximum number of cached results.
        """
        self.max_workers = max_workers
        self.config = config or InferenceConfig()
        self.profile = profile
        self.enable_cache = enable_cache
        self.cache_max_size = cache_max_size

        if profile:
            self.config = profile.get_effective_config()

        # Detection cache: hash -> InferenceResult
        self._cache: dict[str, InferenceResult] = {}

    def detect_batch(
        self,
        items: list[BatchDetectionItem],
        show_progress: bool = False,
    ) -> list[BatchDetectionResult]:
        """Process multiple detection items in parallel.

        Args:
            items: List of BatchDetectionItem to process.
            show_progress: Whether to log progress updates.

        Returns:
            List of BatchDetectionResult in the same order as input.
        """
        if not items:
            return []

        start_time = time.time()
        logger.info(f"Starting batch detection for {len(items)} items")

        # Prepare result placeholders
        results: list[BatchDetectionResult | None] = [None] * len(items)

        # Check cache first
        items_to_process = []
        for i, item in enumerate(items):
            if self.enable_cache:
                cache_key = self._compute_cache_key(item)
                if cache_key in self._cache:
                    results[i] = BatchDetectionResult(
                        item=item,
                        result=self._cache[cache_key],
                        processing_time_ms=0.0,
                    )
                    continue

            items_to_process.append((i, item))

        cache_hits = len(items) - len(items_to_process)
        if cache_hits > 0:
            logger.info(f"Cache hits: {cache_hits}/{len(items)}")

        # Process remaining items in parallel
        if items_to_process:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for i, item in items_to_process:
                    future = executor.submit(self._process_item, item)
                    futures[future] = i

                completed = 0
                for future in as_completed(futures):
                    index = futures[future]
                    item = items[index]

                    try:
                        result = future.result()
                        results[index] = result

                        # Cache successful results
                        if self.enable_cache and result.result and not result.error:
                            cache_key = self._compute_cache_key(item)
                            self._add_to_cache(cache_key, result.result)

                    except Exception as e:
                        logger.error(f"Item {index} failed: {e}")
                        results[index] = BatchDetectionResult(
                            item=item,
                            result=None,
                            error=str(e),
                        )

                    completed += 1
                    if show_progress and completed % 10 == 0:
                        logger.info(f"Progress: {completed}/{len(items_to_process)}")

        elapsed = time.time() - start_time
        logger.info(
            f"Batch detection completed: {len(items)} items in {elapsed:.2f}s "
            f"({len(items) / elapsed:.1f} items/sec)"
        )

        return [r for r in results if r is not None]

    def detect_frames_at_locations(
        self,
        frames: list[np.ndarray],
        click_locations: list[tuple[int, int]],
    ) -> list[InferenceResult]:
        """Process multiple frames with corresponding click locations.

        Convenience method when frames and clicks are in separate lists.

        Args:
            frames: List of frames (same length as click_locations).
            click_locations: List of (x, y) coordinates.

        Returns:
            List of InferenceResult objects.
        """
        if len(frames) != len(click_locations):
            raise ValueError(
                f"Frames and click_locations must have same length: "
                f"{len(frames)} != {len(click_locations)}"
            )

        items = [
            BatchDetectionItem(
                frame=frame,
                click_location=location,
                frame_id=str(i),
            )
            for i, (frame, location) in enumerate(zip(frames, click_locations, strict=True))
        ]

        results = self.detect_batch(items)

        # Extract just the InferenceResult objects
        return [r.result for r in results if r.result is not None]

    def detect_single_frame_multiple_clicks(
        self,
        frame: np.ndarray,
        click_locations: list[tuple[int, int]],
    ) -> list[InferenceResult]:
        """Detect boundaries for multiple clicks on the same frame.

        Since all clicks are on the same frame, this optimizes by
        sharing preprocessing work.

        Args:
            frame: The frame to process.
            click_locations: List of (x, y) click coordinates.

        Returns:
            List of InferenceResult objects.
        """
        if not click_locations:
            return []

        # Create boundary finder once for this frame
        boundary_finder = ElementBoundaryFinder(self.config)
        context_analyzer = ClickContextAnalyzer()

        results = []

        for click_location in click_locations:
            start_time = time.time()

            click_x, click_y = click_location
            height, width = frame.shape[:2]

            if not (0 <= click_x < width and 0 <= click_y < height):
                # Create fallback result
                result = self._create_fallback_result(click_location, width, height)
            else:
                # Find boundaries
                candidates = boundary_finder.find_boundaries(
                    frame, click_location, self.config.preferred_strategies
                )

                if candidates:
                    for candidate in candidates:
                        if self.config.enable_element_classification:
                            element_type, confidence = context_analyzer.get_element_type_confidence(
                                frame, candidate, click_location
                            )
                            candidate.element_type = element_type
                            candidate.confidence = (candidate.confidence + confidence) / 2

                    candidates.sort(key=lambda c: -c.confidence)

                    processing_time = (time.time() - start_time) * 1000

                    result = InferenceResult(
                        click_location=click_location,
                        primary_bbox=candidates[0],
                        alternative_candidates=candidates[1:5],
                        image_width=width,
                        image_height=height,
                        strategies_attempted=self.config.preferred_strategies,
                        processing_time_ms=processing_time,
                        used_fallback=False,
                    )
                else:
                    result = self._create_fallback_result(click_location, width, height)

            results.append(result)

        return results

    def _process_item(self, item: BatchDetectionItem) -> BatchDetectionResult:
        """Process a single detection item."""
        start_time = time.time()

        try:
            # Create fresh finder for this thread
            boundary_finder = ElementBoundaryFinder(self.config)
            context_analyzer = ClickContextAnalyzer()

            frame = item.frame
            click_location = item.click_location

            click_x, click_y = click_location
            height, width = frame.shape[:2]

            if not (0 <= click_x < width and 0 <= click_y < height):
                result = self._create_fallback_result(click_location, width, height)
            else:
                candidates = boundary_finder.find_boundaries(
                    frame, click_location, self.config.preferred_strategies
                )

                if candidates:
                    for candidate in candidates:
                        if self.config.enable_element_classification:
                            element_type, confidence = context_analyzer.get_element_type_confidence(
                                frame, candidate, click_location
                            )
                            candidate.element_type = element_type
                            candidate.confidence = (candidate.confidence + confidence) / 2

                    candidates.sort(key=lambda c: -c.confidence)

                    processing_time = (time.time() - start_time) * 1000

                    result = InferenceResult(
                        click_location=click_location,
                        primary_bbox=candidates[0],
                        alternative_candidates=candidates[1:5],
                        image_width=width,
                        image_height=height,
                        strategies_attempted=self.config.preferred_strategies,
                        processing_time_ms=processing_time,
                        used_fallback=False,
                    )
                else:
                    result = self._create_fallback_result(click_location, width, height)

            processing_time = (time.time() - start_time) * 1000

            return BatchDetectionResult(
                item=item,
                result=result,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Detection failed for item: {e}")

            return BatchDetectionResult(
                item=item,
                result=None,
                processing_time_ms=processing_time,
                error=str(e),
            )

    def _create_fallback_result(
        self,
        click_location: tuple[int, int],
        img_width: int,
        img_height: int,
    ) -> InferenceResult:
        """Create a fallback result with fixed-size bounding box."""
        from .models import ElementType

        click_x, click_y = click_location
        size = self.config.fallback_box_size
        half_size = size // 2

        x = max(0, click_x - half_size)
        y = max(0, click_y - half_size)
        w = min(size, img_width - x)
        h = min(size, img_height - y)

        fallback_bbox = InferredBoundingBox(
            x=x,
            y=y,
            width=w,
            height=h,
            confidence=0.3,
            strategy_used=DetectionStrategy.FIXED_SIZE,
            element_type=ElementType.UNKNOWN,
            metadata={"fallback": True},
        )

        return InferenceResult(
            click_location=click_location,
            primary_bbox=fallback_bbox,
            image_width=img_width,
            image_height=img_height,
            strategies_attempted=[DetectionStrategy.FIXED_SIZE],
            processing_time_ms=0.0,
            used_fallback=True,
        )

    def _compute_cache_key(self, item: BatchDetectionItem) -> str:
        """Compute a cache key for a detection item."""
        # Hash based on click location and a sample of pixels around it
        click_x, click_y = item.click_location
        frame = item.frame
        height, width = frame.shape[:2]

        # Sample a small region around the click
        sample_size = 32
        x1 = max(0, click_x - sample_size)
        y1 = max(0, click_y - sample_size)
        x2 = min(width, click_x + sample_size)
        y2 = min(height, click_y + sample_size)

        region = frame[y1:y2, x1:x2]

        # Create hash
        hasher = hashlib.md5()
        hasher.update(f"{click_x},{click_y}".encode())
        hasher.update(region.tobytes())

        return hasher.hexdigest()

    def _add_to_cache(self, key: str, result: InferenceResult) -> None:
        """Add a result to the cache, evicting old entries if needed."""
        if len(self._cache) >= self.cache_max_size:
            # Simple FIFO eviction: remove first 10%
            keys_to_remove = list(self._cache.keys())[: self.cache_max_size // 10]
            for k in keys_to_remove:
                del self._cache[k]

        self._cache[key] = result

    def clear_cache(self) -> None:
        """Clear the detection cache."""
        self._cache.clear()
        logger.info("Detection cache cleared")

    @property
    def cache_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
