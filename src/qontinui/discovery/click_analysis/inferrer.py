"""Main click bounding box inferrer that orchestrates detection strategies."""

import logging
import time
from typing import Any

import numpy as np

from .boundary_finder import ElementBoundaryFinder
from .context_analyzer import ClickContextAnalyzer
from .models import (
    DetectionStrategy,
    ElementType,
    InferenceConfig,
    InferenceResult,
    InferredBoundingBox,
)

logger = logging.getLogger(__name__)


class ClickBoundingBoxInferrer:
    """Main class for inferring bounding boxes from click locations.

    This class orchestrates the detection process:
    1. Uses ElementBoundaryFinder to detect potential element boundaries
    2. Uses ClickContextAnalyzer to determine element types
    3. Ranks candidates and selects the best match
    4. Provides fallback to fixed-size box if needed

    Example:
        >>> inferrer = ClickBoundingBoxInferrer()
        >>> result = inferrer.infer_bounding_box(
        ...     screenshot=screenshot_array,
        ...     click_location=(350, 250)
        ... )
        >>> bbox = result.primary_bbox
        >>> print(f"Detected {bbox.element_type.value} at ({bbox.x}, {bbox.y})")
    """

    def __init__(self, config: InferenceConfig | None = None) -> None:
        """Initialize the inferrer.

        Args:
            config: Configuration for inference. Uses defaults if not provided.
        """
        self.config = config or InferenceConfig()
        self.boundary_finder = ElementBoundaryFinder(self.config)
        self.context_analyzer = ClickContextAnalyzer()

    def infer_bounding_box(
        self,
        screenshot: np.ndarray[Any, Any],
        click_location: tuple[int, int],
        existing_state_images: list[Any] | None = None,
    ) -> InferenceResult:
        """Infer the bounding box of the element at a click location.

        Args:
            screenshot: Screenshot image (BGR or RGB format).
            click_location: (x, y) coordinates of the click.
            existing_state_images: Optional list of known StateImages to check against.

        Returns:
            InferenceResult with the detected bounding box and metadata.
        """
        start_time = time.time()

        click_x, click_y = click_location
        height, width = screenshot.shape[:2]

        # Validate click location
        if not (0 <= click_x < width and 0 <= click_y < height):
            logger.warning(
                f"Click location ({click_x}, {click_y}) outside image bounds ({width}x{height})"
            )
            # Return fallback centered at nearest valid point
            valid_x = max(0, min(click_x, width - 1))
            valid_y = max(0, min(click_y, height - 1))
            return self._create_fallback_result((valid_x, valid_y), width, height, start_time, [])

        strategies_attempted: list[DetectionStrategy] = []

        # Step 1: Check against existing state images if provided
        if existing_state_images:
            match = self._check_existing_state_images(
                screenshot, click_location, existing_state_images
            )
            if match:
                strategies_attempted.append(DetectionStrategy.TEMPLATE_MATCH)
                processing_time = (time.time() - start_time) * 1000
                return InferenceResult(
                    click_location=click_location,
                    primary_bbox=match,
                    image_width=width,
                    image_height=height,
                    strategies_attempted=strategies_attempted,
                    processing_time_ms=processing_time,
                    used_fallback=False,
                )

        # Step 2: Find element boundaries using multiple strategies
        candidates = self.boundary_finder.find_boundaries(
            screenshot, click_location, self.config.preferred_strategies
        )

        strategies_attempted.extend(
            [s for s in self.config.preferred_strategies if s != DetectionStrategy.FIXED_SIZE]
        )

        # Step 3: If candidates found, classify and rank them
        if candidates:
            # Classify element types
            for candidate in candidates:
                if self.config.enable_element_classification:
                    element_type, type_confidence = (
                        self.context_analyzer.get_element_type_confidence(
                            screenshot, candidate, click_location
                        )
                    )
                    candidate.element_type = element_type
                    # Blend type confidence into overall confidence
                    candidate.confidence = (candidate.confidence + type_confidence) / 2
                    candidate.metadata["type_confidence"] = type_confidence

            # Sort by confidence
            candidates.sort(key=lambda c: -c.confidence)

            primary = candidates[0]
            alternatives = candidates[1:5]  # Keep top 5 alternatives

            processing_time = (time.time() - start_time) * 1000

            return InferenceResult(
                click_location=click_location,
                primary_bbox=primary,
                alternative_candidates=alternatives,
                image_width=width,
                image_height=height,
                strategies_attempted=strategies_attempted,
                processing_time_ms=processing_time,
                used_fallback=False,
            )

        # Step 4: Fallback to fixed-size box
        if self.config.use_fallback:
            return self._create_fallback_result(
                click_location, width, height, start_time, strategies_attempted
            )

        # No detection and fallback disabled
        processing_time = (time.time() - start_time) * 1000
        return InferenceResult(
            click_location=click_location,
            primary_bbox=self._create_minimal_bbox(click_location, width, height),
            image_width=width,
            image_height=height,
            strategies_attempted=strategies_attempted,
            processing_time_ms=processing_time,
            used_fallback=True,
        )

    def infer_bbox_simple(
        self,
        screenshot: np.ndarray[Any, Any],
        click_location: tuple[int, int],
    ) -> list[int]:
        """Simplified interface returning just [x, y, width, height].

        This is a convenience method for simple use cases where only
        the bounding box coordinates are needed.

        Args:
            screenshot: Screenshot image.
            click_location: (x, y) coordinates of the click.

        Returns:
            List of [x, y, width, height].
        """
        result = self.infer_bounding_box(screenshot, click_location)
        return result.primary_bbox.as_bbox_list()

    def _check_existing_state_images(
        self,
        screenshot: np.ndarray[Any, Any],
        click_location: tuple[int, int],
        state_images: list[Any],
    ) -> InferredBoundingBox | None:
        """Check if click matches any existing StateImage.

        Args:
            screenshot: Screenshot image.
            click_location: Click coordinates.
            state_images: List of StateImage objects.

        Returns:
            InferredBoundingBox if match found, None otherwise.
        """
        click_x, click_y = click_location

        for state_image in state_images:
            # Check if click is within state image bounds
            if not state_image.contains_point(click_x, click_y):
                continue

            # Verify the state image is present in the screenshot
            if not self._is_state_image_present(screenshot, state_image):
                continue

            # Create bbox from state image
            return InferredBoundingBox(
                x=state_image.x,
                y=state_image.y,
                width=state_image.width,
                height=state_image.height,
                confidence=0.95,  # High confidence for known elements
                strategy_used=DetectionStrategy.TEMPLATE_MATCH,
                element_type=ElementType.UNKNOWN,  # Could be inferred from state_image tags
                mask=state_image.mask,
                pixel_data=state_image.pixel_data,
                metadata={
                    "state_image_id": state_image.id,
                    "state_image_name": state_image.name,
                    "source": "existing_state_image",
                },
            )

        return None

    def _is_state_image_present(
        self,
        screenshot: np.ndarray[Any, Any],
        state_image: Any,
    ) -> bool:
        """Check if a StateImage is present at its location in the screenshot."""
        x, y = state_image.x, state_image.y
        x2, y2 = state_image.x2, state_image.y2

        height, width = screenshot.shape[:2]
        if x2 > width or y2 > height:
            return False

        roi = screenshot[y : y2 + 1, x : x2 + 1]

        if state_image.pixel_data is None:
            return True  # Can't verify without pixel data

        if roi.shape != state_image.pixel_data.shape:
            return False

        # Calculate similarity
        diff = np.abs(roi.astype(np.float32) - state_image.pixel_data.astype(np.float32))

        if state_image.mask is not None:
            # Apply mask
            mask = state_image.mask
            if len(mask.shape) == 2 and len(diff.shape) == 3:
                mask = np.expand_dims(mask, axis=2)
            diff = diff * mask
            active_pixels = np.sum(mask > 0.5)
            if active_pixels == 0:
                return True
            mean_diff = np.sum(diff) / (active_pixels * 3 * 255)
        else:
            mean_diff = np.mean(diff) / 255

        similarity = 1.0 - mean_diff
        return similarity >= 0.9  # 90% similarity threshold

    def _create_fallback_result(
        self,
        click_location: tuple[int, int],
        img_width: int,
        img_height: int,
        start_time: float,
        strategies_attempted: list[DetectionStrategy],
    ) -> InferenceResult:
        """Create a fallback result with fixed-size bounding box."""
        fallback_bbox = self._create_fallback_bbox(click_location, img_width, img_height)

        strategies_attempted.append(DetectionStrategy.FIXED_SIZE)
        processing_time = (time.time() - start_time) * 1000

        return InferenceResult(
            click_location=click_location,
            primary_bbox=fallback_bbox,
            image_width=img_width,
            image_height=img_height,
            strategies_attempted=strategies_attempted,
            processing_time_ms=processing_time,
            used_fallback=True,
        )

    def _create_fallback_bbox(
        self,
        click_location: tuple[int, int],
        img_width: int,
        img_height: int,
    ) -> InferredBoundingBox:
        """Create a fixed-size bounding box centered on click."""
        click_x, click_y = click_location
        size = self.config.fallback_box_size
        half_size = size // 2

        # Center on click, clamp to image bounds
        x = max(0, click_x - half_size)
        y = max(0, click_y - half_size)
        w = min(size, img_width - x)
        h = min(size, img_height - y)

        return InferredBoundingBox(
            x=x,
            y=y,
            width=w,
            height=h,
            confidence=0.3,  # Low confidence for fallback
            strategy_used=DetectionStrategy.FIXED_SIZE,
            element_type=ElementType.UNKNOWN,
            metadata={"fallback": True, "original_click": click_location},
        )

    def _create_minimal_bbox(
        self,
        click_location: tuple[int, int],
        img_width: int,
        img_height: int,
    ) -> InferredBoundingBox:
        """Create a minimal bounding box at click point."""
        click_x, click_y = click_location

        return InferredBoundingBox(
            x=max(0, click_x - 5),
            y=max(0, click_y - 5),
            width=min(10, img_width),
            height=min(10, img_height),
            confidence=0.1,
            strategy_used=DetectionStrategy.FIXED_SIZE,
            element_type=ElementType.UNKNOWN,
            metadata={"minimal": True},
        )


# Convenience function for simple usage
def infer_bbox_from_click(
    screenshot: np.ndarray[Any, Any],
    click_location: tuple[int, int],
    config: InferenceConfig | None = None,
) -> InferenceResult:
    """Convenience function to infer bounding box from click.

    Args:
        screenshot: Screenshot image (BGR or RGB format).
        click_location: (x, y) coordinates of the click.
        config: Optional configuration.

    Returns:
        InferenceResult with detected bounding box.

    Example:
        >>> from qontinui.discovery.click_analysis import infer_bbox_from_click
        >>> result = infer_bbox_from_click(screenshot, (350, 250))
        >>> bbox = result.primary_bbox.as_bbox_list()
    """
    inferrer = ClickBoundingBoxInferrer(config)
    return inferrer.infer_bounding_box(screenshot, click_location)
