"""Element boundary detection using multiple strategies."""

import logging
from typing import Any

import cv2
import numpy as np

from .models import DetectionStrategy, InferenceConfig, InferredBoundingBox

logger = logging.getLogger(__name__)


class ElementBoundaryFinder:
    """Finds element boundaries around a click point using multiple strategies.

    This class implements several strategies for detecting the boundaries of
    GUI elements at a given click location:

    1. Contour-based: Finds closed contours containing the click point
    2. Edge-based: Uses Canny edge detection to find element borders
    3. Color segmentation: Groups pixels by color similarity
    4. Flood fill: Expands from click point to find uniform regions
    5. Gradient-based: Detects boundaries via intensity gradients

    Each strategy returns candidates ranked by confidence.
    """

    def __init__(self, config: InferenceConfig | None = None) -> None:
        """Initialize the boundary finder.

        Args:
            config: Configuration for detection parameters.
        """
        self.config = config or InferenceConfig()

    def find_boundaries(
        self,
        screenshot: np.ndarray[Any, Any],
        click_location: tuple[int, int],
        strategies: list[DetectionStrategy] | None = None,
    ) -> list[InferredBoundingBox]:
        """Find element boundaries using specified strategies.

        Args:
            screenshot: Screenshot image (BGR or RGB format).
            click_location: (x, y) coordinates of the click.
            strategies: List of strategies to try. Defaults to config.preferred_strategies.

        Returns:
            List of InferredBoundingBox candidates, sorted by confidence.
        """
        if strategies is None:
            strategies = self.config.preferred_strategies

        click_x, click_y = click_location
        height, width = screenshot.shape[:2]

        # Validate click is within image
        if not (0 <= click_x < width and 0 <= click_y < height):
            logger.warning(f"Click location ({click_x}, {click_y}) outside image bounds")
            return []

        candidates: list[InferredBoundingBox] = []

        for strategy in strategies:
            if strategy == DetectionStrategy.FIXED_SIZE:
                # Skip fixed size - it's the fallback handled elsewhere
                continue

            try:
                strategy_candidates = self._apply_strategy(screenshot, click_location, strategy)
                candidates.extend(strategy_candidates)
            except Exception as e:
                logger.warning(f"Strategy {strategy.value} failed: {e}")

        # Sort by confidence (descending) and distance from click (ascending)
        candidates.sort(
            key=lambda c: (
                -c.confidence,
                self._distance_to_center(c, click_location),
            )
        )

        # Remove duplicates (overlapping boxes)
        candidates = self._remove_duplicates(candidates)

        return candidates

    def _apply_strategy(
        self,
        screenshot: np.ndarray[Any, Any],
        click_location: tuple[int, int],
        strategy: DetectionStrategy,
    ) -> list[InferredBoundingBox]:
        """Apply a single detection strategy."""
        if strategy == DetectionStrategy.CONTOUR_BASED:
            return self._find_by_contour(screenshot, click_location)
        elif strategy == DetectionStrategy.EDGE_BASED:
            return self._find_by_edge(screenshot, click_location)
        elif strategy == DetectionStrategy.COLOR_SEGMENTATION:
            return self._find_by_color(screenshot, click_location)
        elif strategy == DetectionStrategy.FLOOD_FILL:
            return self._find_by_flood_fill(screenshot, click_location)
        elif strategy == DetectionStrategy.GRADIENT_BASED:
            return self._find_by_gradient(screenshot, click_location)
        else:
            return []

    def _find_by_contour(
        self,
        screenshot: np.ndarray[Any, Any],
        click_location: tuple[int, int],
    ) -> list[InferredBoundingBox]:
        """Find element by detecting contours containing the click point."""
        click_x, click_y = click_location
        height, width = screenshot.shape[:2]

        # Convert to grayscale
        if len(screenshot.shape) == 3:
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        else:
            gray = screenshot

        # Apply multiple edge detection passes for robustness
        candidates = []

        for threshold_multiplier in [1.0, 0.7, 1.3]:
            low = int(self.config.edge_threshold_low * threshold_multiplier)
            high = int(self.config.edge_threshold_high * threshold_multiplier)

            edges = cv2.Canny(gray, low, high)

            # Dilate to close small gaps
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            # Find contours
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for _i, contour in enumerate(contours):
                # Check if click point is inside contour
                if cv2.pointPolygonTest(contour, (click_x, click_y), False) < 0:
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                # Check size constraints
                if not self._check_size_constraints(w, h):
                    continue

                # Check distance from click
                if not self._is_within_search_radius(x, y, w, h, click_location):
                    continue

                # Calculate confidence based on:
                # - Contour area vs bounding box area (rectangularity)
                # - Distance from click to center
                # - Whether it's a parent or child contour
                contour_area = cv2.contourArea(contour)
                bbox_area = w * h
                rectangularity = contour_area / bbox_area if bbox_area > 0 else 0

                center_dist = self._distance_to_center(
                    InferredBoundingBox(
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                        confidence=0,
                        strategy_used=DetectionStrategy.CONTOUR_BASED,
                    ),
                    click_location,
                )
                max_dist = self.config.search_radius
                dist_score = 1.0 - (center_dist / max_dist) if max_dist > 0 else 0.5

                # Prefer smaller, well-defined contours containing the click
                size_penalty = min(1.0, 10000 / bbox_area) if bbox_area > 0 else 0

                confidence = 0.3 + rectangularity * 0.3 + dist_score * 0.2 + size_penalty * 0.2

                # Generate mask if enabled
                mask = None
                if self.config.enable_mask_generation:
                    mask = np.zeros((h, w), dtype=np.float32)
                    shifted_contour = contour - [x, y]
                    cv2.drawContours(mask, [shifted_contour], -1, 1.0, -1)

                # Extract pixel data
                pixel_data = screenshot[y : y + h, x : x + w].copy()

                candidates.append(
                    InferredBoundingBox(
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                        confidence=min(0.95, confidence),
                        strategy_used=DetectionStrategy.CONTOUR_BASED,
                        mask=mask,
                        pixel_data=pixel_data,
                        metadata={
                            "rectangularity": rectangularity,
                            "contour_area": int(contour_area),
                        },
                    )
                )

        return candidates

    def _find_by_edge(
        self,
        screenshot: np.ndarray[Any, Any],
        click_location: tuple[int, int],
    ) -> list[InferredBoundingBox]:
        """Find element using edge detection to locate boundaries."""
        click_x, click_y = click_location
        height, width = screenshot.shape[:2]

        # Define search region
        radius = self.config.search_radius
        x1 = max(0, click_x - radius)
        y1 = max(0, click_y - radius)
        x2 = min(width, click_x + radius)
        y2 = min(height, click_y + radius)

        roi = screenshot[y1:y2, x1:x2]

        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # Detect edges
        edges = cv2.Canny(
            gray,
            self.config.edge_threshold_low,
            self.config.edge_threshold_high,
        )

        # Find boundaries by scanning from click point
        local_click_x = click_x - x1
        local_click_y = click_y - y1

        # Scan in 4 directions to find edges
        left_edge = self._scan_for_edge(edges, local_click_x, local_click_y, -1, 0)
        right_edge = self._scan_for_edge(edges, local_click_x, local_click_y, 1, 0)
        top_edge = self._scan_for_edge(edges, local_click_x, local_click_y, 0, -1)
        bottom_edge = self._scan_for_edge(edges, local_click_x, local_click_y, 0, 1)

        # Convert back to image coordinates
        bbox_x = x1 + left_edge
        bbox_y = y1 + top_edge
        bbox_w = right_edge - left_edge + 1
        bbox_h = bottom_edge - top_edge + 1

        if not self._check_size_constraints(bbox_w, bbox_h):
            return []

        # Calculate confidence based on edge strength
        edge_roi = edges[top_edge : bottom_edge + 1, left_edge : right_edge + 1]
        edge_density = np.sum(edge_roi > 0) / edge_roi.size if edge_roi.size > 0 else 0

        confidence = 0.4 + edge_density * 0.4

        pixel_data = screenshot[bbox_y : bbox_y + bbox_h, bbox_x : bbox_x + bbox_w].copy()

        return [
            InferredBoundingBox(
                x=bbox_x,
                y=bbox_y,
                width=bbox_w,
                height=bbox_h,
                confidence=min(0.85, confidence),
                strategy_used=DetectionStrategy.EDGE_BASED,
                pixel_data=pixel_data,
                metadata={"edge_density": float(edge_density)},
            )
        ]

    def _find_by_color(
        self,
        screenshot: np.ndarray[Any, Any],
        click_location: tuple[int, int],
    ) -> list[InferredBoundingBox]:
        """Find element by grouping similar colored pixels around click."""
        click_x, click_y = click_location
        height, width = screenshot.shape[:2]

        # Get reference color at click point
        if len(screenshot.shape) == 3:
            ref_color = screenshot[click_y, click_x].astype(np.int32)
        else:
            ref_color = np.array([screenshot[click_y, click_x]] * 3, dtype=np.int32)

        # Define search region
        radius = self.config.search_radius
        x1 = max(0, click_x - radius)
        y1 = max(0, click_y - radius)
        x2 = min(width, click_x + radius)
        y2 = min(height, click_y + radius)

        roi = screenshot[y1:y2, x1:x2]

        # Create mask of similar colors
        if len(roi.shape) == 3:
            diff = np.abs(roi.astype(np.int32) - ref_color)
            color_mask = np.all(diff < self.config.color_tolerance, axis=2).astype(np.uint8)
        else:
            diff = np.abs(roi.astype(np.int32) - ref_color[0])
            color_mask = (diff < self.config.color_tolerance).astype(np.uint8)

        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the color mask
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        local_click = (click_x - x1, click_y - y1)

        for contour in contours:
            # Check if click point is inside
            if cv2.pointPolygonTest(contour, local_click, False) < 0:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Convert to image coordinates
            bbox_x = x1 + x
            bbox_y = y1 + y

            if not self._check_size_constraints(w, h):
                continue

            # Calculate confidence
            contour_area = cv2.contourArea(contour)
            bbox_area = w * h
            fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0

            confidence = 0.5 + fill_ratio * 0.4

            # Generate mask
            mask = None
            if self.config.enable_mask_generation:
                mask = np.zeros((h, w), dtype=np.float32)
                shifted_contour = contour - [x, y]
                cv2.drawContours(mask, [shifted_contour], -1, 1.0, -1)

            pixel_data = screenshot[bbox_y : bbox_y + h, bbox_x : bbox_x + w].copy()

            candidates.append(
                InferredBoundingBox(
                    x=bbox_x,
                    y=bbox_y,
                    width=w,
                    height=h,
                    confidence=min(0.9, confidence),
                    strategy_used=DetectionStrategy.COLOR_SEGMENTATION,
                    mask=mask,
                    pixel_data=pixel_data,
                    metadata={
                        "fill_ratio": float(fill_ratio),
                        "reference_color": ref_color.tolist(),
                    },
                )
            )

        return candidates

    def _find_by_flood_fill(
        self,
        screenshot: np.ndarray[Any, Any],
        click_location: tuple[int, int],
    ) -> list[InferredBoundingBox]:
        """Find element using flood fill from click point."""
        click_x, click_y = click_location
        height, width = screenshot.shape[:2]

        # Work on a copy
        if len(screenshot.shape) == 3:
            img = screenshot.copy()
        else:
            img = cv2.cvtColor(screenshot, cv2.COLOR_GRAY2BGR)

        # Create mask for flood fill (needs to be 2 pixels larger)
        mask = np.zeros((height + 2, width + 2), np.uint8)

        # Flood fill with tolerance
        tolerance = self.config.color_tolerance
        lo_diff = (tolerance, tolerance, tolerance)
        hi_diff = (tolerance, tolerance, tolerance)

        # Use a unique fill color
        fill_color = (255, 0, 255)

        # Flood fill
        retval, img_filled, mask_filled, rect = cv2.floodFill(
            img,
            mask,
            (click_x, click_y),
            fill_color,
            lo_diff,
            hi_diff,
            cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE,
        )

        if retval == 0:
            return []

        # Extract bounding box from rect
        x, y, w, h = rect

        if not self._check_size_constraints(w, h):
            return []

        # Extract the actual filled mask (remove padding)
        element_mask = mask_filled[1:-1, 1:-1]
        element_mask = element_mask[y : y + h, x : x + w]

        # Calculate confidence based on fill area
        fill_area = np.sum(element_mask > 0)
        bbox_area = w * h
        fill_ratio = fill_area / bbox_area if bbox_area > 0 else 0

        confidence = 0.5 + fill_ratio * 0.35

        # Convert mask to float
        mask_float = None
        if self.config.enable_mask_generation:
            mask_float = element_mask.astype(np.float32)

        pixel_data = screenshot[y : y + h, x : x + w].copy()

        return [
            InferredBoundingBox(
                x=x,
                y=y,
                width=w,
                height=h,
                confidence=min(0.85, confidence),
                strategy_used=DetectionStrategy.FLOOD_FILL,
                mask=mask_float,
                pixel_data=pixel_data,
                metadata={
                    "fill_area": int(fill_area),
                    "fill_ratio": float(fill_ratio),
                },
            )
        ]

    def _find_by_gradient(
        self,
        screenshot: np.ndarray[Any, Any],
        click_location: tuple[int, int],
    ) -> list[InferredBoundingBox]:
        """Find element by detecting gradient boundaries."""
        click_x, click_y = click_location
        height, width = screenshot.shape[:2]

        # Convert to grayscale
        if len(screenshot.shape) == 3:
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        else:
            gray = screenshot

        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold gradient
        grad_thresh = np.percentile(gradient_magnitude, 80)
        grad_binary = (gradient_magnitude > grad_thresh).astype(np.uint8) * 255

        # Search region
        radius = self.config.search_radius
        x1 = max(0, click_x - radius)
        y1 = max(0, click_y - radius)
        x2 = min(width, click_x + radius)
        y2 = min(height, click_y + radius)

        grad_roi = grad_binary[y1:y2, x1:x2]

        # Find contours in gradient
        contours, _ = cv2.findContours(grad_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        local_click = (click_x - x1, click_y - y1)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Check if click is within or near this contour
            if not (x <= local_click[0] <= x + w and y <= local_click[1] <= y + h):
                continue

            bbox_x = x1 + x
            bbox_y = y1 + y

            if not self._check_size_constraints(w, h):
                continue

            # Confidence based on gradient strength in region
            region_grad = gradient_magnitude[bbox_y : bbox_y + h, bbox_x : bbox_x + w]
            mean_grad = np.mean(region_grad)
            max_grad = np.max(gradient_magnitude)
            grad_score = mean_grad / max_grad if max_grad > 0 else 0

            confidence = 0.4 + grad_score * 0.4

            pixel_data = screenshot[bbox_y : bbox_y + h, bbox_x : bbox_x + w].copy()

            candidates.append(
                InferredBoundingBox(
                    x=bbox_x,
                    y=bbox_y,
                    width=w,
                    height=h,
                    confidence=min(0.8, confidence),
                    strategy_used=DetectionStrategy.GRADIENT_BASED,
                    pixel_data=pixel_data,
                    metadata={"gradient_score": float(grad_score)},
                )
            )

        return candidates

    def _scan_for_edge(
        self,
        edges: np.ndarray[Any, Any],
        start_x: int,
        start_y: int,
        dx: int,
        dy: int,
    ) -> int:
        """Scan from a point in a direction until an edge is found."""
        height, width = edges.shape
        x, y = start_x, start_y
        max_steps = self.config.search_radius

        for _ in range(max_steps):
            x += dx
            y += dy

            if x < 0 or x >= width or y < 0 or y >= height:
                break

            if edges[y, x] > 0:
                break

        # Return the position (accounting for direction)
        if dx != 0:
            return x
        else:
            return y

    def _check_size_constraints(self, w: int, h: int) -> bool:
        """Check if dimensions meet size constraints."""
        min_w, min_h = self.config.min_element_size
        max_w, max_h = self.config.max_element_size
        return min_w <= w <= max_w and min_h <= h <= max_h

    def _is_within_search_radius(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        click_location: tuple[int, int],
    ) -> bool:
        """Check if bounding box is within search radius of click."""
        click_x, click_y = click_location
        center_x = x + w // 2
        center_y = y + h // 2
        dist = np.sqrt((center_x - click_x) ** 2 + (center_y - click_y) ** 2)
        return dist <= self.config.search_radius

    def _distance_to_center(
        self,
        bbox: InferredBoundingBox,
        click_location: tuple[int, int],
    ) -> float:
        """Calculate distance from click to bbox center."""
        center_x, center_y = bbox.center
        click_x, click_y = click_location
        return float(np.sqrt((center_x - click_x) ** 2 + (center_y - click_y) ** 2))

    def _remove_duplicates(
        self,
        candidates: list[InferredBoundingBox],
        iou_threshold: float = 0.5,
    ) -> list[InferredBoundingBox]:
        """Remove duplicate/overlapping candidates using NMS-like approach."""
        if not candidates:
            return []

        keep = []

        for candidate in candidates:
            is_duplicate = False

            for kept in keep:
                iou = self._calculate_iou(candidate, kept)
                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep.append(candidate)

        return keep

    def _calculate_iou(
        self,
        box1: InferredBoundingBox,
        box2: InferredBoundingBox,
    ) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = box1.area + box2.area - intersection

        return intersection / union if union > 0 else 0.0
