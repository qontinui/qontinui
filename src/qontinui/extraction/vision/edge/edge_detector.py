"""
Edge Detection for StateImage Candidate Discovery.

Uses classical computer vision (Canny edge detection + contour analysis)
to detect UI element boundaries in screenshots.
"""

import base64
import logging
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from ..models import (
    BoundingBox,
    ContourResult,
    EdgeDetectionConfig,
    EdgeDetectionResult,
    ExtractedStateImageCandidate,
)

logger = logging.getLogger(__name__)


class EdgeDetector:
    """
    Classical edge detection for UI element boundaries.

    Uses Canny edge detection to find edges, then finds contours
    and filters them by size, aspect ratio, and shape properties.
    """

    def __init__(self, config: EdgeDetectionConfig | None = None) -> None:
        """
        Initialize the edge detector.

        Args:
            config: Edge detection configuration. Uses defaults if not provided.
        """
        self.config = config or EdgeDetectionConfig()
        self._result_counter = 0

    def detect(
        self,
        screenshot: np.ndarray,
        screenshot_id: str,
    ) -> tuple[list[EdgeDetectionResult], list[ContourResult], np.ndarray | None]:
        """
        Detect element boundaries using edge detection.

        Args:
            screenshot: BGR image as numpy array.
            screenshot_id: ID of the screenshot for reference.

        Returns:
            Tuple of:
                - List of EdgeDetectionResult for filtered detections
                - List of ContourResult for all contours (debug)
                - Edge overlay image (debug visualization)
        """
        if not self.config.enabled:
            return [], [], None

        logger.info("Running edge detection...")

        # Convert to grayscale
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(
            blurred,
            self.config.canny_low,
            self.config.canny_high,
        )

        # Morphological operations to close gaps
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Process contours
        edge_results: list[EdgeDetectionResult] = []
        contour_results: list[ContourResult] = []

        if hierarchy is not None:
            hierarchy = hierarchy[0]  # Unwrap

        for idx, contour in enumerate(contours):
            # Get contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate hierarchy level
            h_level = 0
            parent_idx = None
            if hierarchy is not None:
                parent_idx = hierarchy[idx][3]
                # Count levels up to root
                current = idx
                while hierarchy[current][3] != -1:
                    h_level += 1
                    current = hierarchy[current][3]

            # Create ContourResult for debug
            self._result_counter += 1
            contour_id = f"contour_{self._result_counter:06d}"

            contour_result = ContourResult(
                id=contour_id,
                bbox=BoundingBox(x=x, y=y, width=w, height=h),
                area=area,
                perimeter=perimeter,
                hierarchy_level=h_level,
                parent_id=(
                    f"contour_{parent_idx:06d}"
                    if parent_idx is not None and parent_idx >= 0
                    else None
                ),
            )
            contour_results.append(contour_result)

            # Filter by area
            if area < self.config.min_contour_area:
                continue
            if area > self.config.max_contour_area:
                continue

            # Filter by aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < self.config.aspect_ratio_min:
                continue
            if aspect_ratio > self.config.aspect_ratio_max:
                continue

            # Approximate to polygon
            epsilon = self.config.approximation_epsilon * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertex_count = len(approx)

            # Calculate confidence based on shape regularity
            # Rectangles (4 vertices) get higher confidence
            if vertex_count == 4:
                confidence = 0.9
            elif vertex_count <= 6:
                confidence = 0.7
            else:
                confidence = 0.5

            # Adjust confidence by area (larger areas more likely to be real elements)
            area_factor = min(area / 10000, 1.0)  # Normalize to [0, 1]
            confidence = confidence * 0.7 + area_factor * 0.3

            # Create EdgeDetectionResult
            edge_result = EdgeDetectionResult(
                id=contour_id,
                bbox=BoundingBox(x=x, y=y, width=w, height=h),
                contour_area=area,
                contour_perimeter=perimeter,
                vertex_count=vertex_count,
                aspect_ratio=aspect_ratio,
                confidence=confidence,
                contour_points=[(int(p[0][0]), int(p[0][1])) for p in approx],
            )
            edge_results.append(edge_result)

        # Create debug overlay image
        overlay = self._create_edge_overlay(screenshot, edges, edge_results)

        logger.info(
            f"Edge detection found {len(contour_results)} contours, "
            f"{len(edge_results)} passed filters"
        )

        return edge_results, contour_results, overlay

    def results_to_candidates(
        self,
        edge_results: list[EdgeDetectionResult],
        screenshot_id: str,
    ) -> list[ExtractedStateImageCandidate]:
        """
        Convert edge detection results to StateImage candidates.

        Args:
            edge_results: Results from edge detection.
            screenshot_id: ID of the source screenshot.

        Returns:
            List of ExtractedStateImageCandidate.
        """
        candidates = []

        for result in edge_results:
            # Classify shape based on vertex count and aspect ratio
            category = self._classify_shape(result)

            candidate = ExtractedStateImageCandidate(
                id=f"edge_{result.id}",
                bbox=result.bbox,
                confidence=result.confidence,
                screenshot_id=screenshot_id,
                category=category,
                detection_technique="edge",
                is_clickable=(category in ("button", "input", "link")),
                metadata={
                    "contour_area": result.contour_area,
                    "vertex_count": result.vertex_count,
                    "aspect_ratio": result.aspect_ratio,
                },
            )
            candidates.append(candidate)

        return candidates

    def _classify_shape(self, result: EdgeDetectionResult) -> str:
        """
        Classify detected shape as element category.

        This is for description only - categories don't have
        functional significance in the state machine.
        """
        bbox = result.bbox
        aspect_ratio = result.aspect_ratio
        vertex_count = result.vertex_count

        # Check if rectangular (4 vertices)
        if vertex_count == 4:
            # Button-like: rectangular, moderate aspect ratio
            if 1.5 < aspect_ratio < 6.0 and 30 < bbox.width < 300 and 20 < bbox.height < 60:
                return "button"
            # Input field: wide and short
            if aspect_ratio > 4.0 and bbox.height < 50:
                return "input"
            # Could be a card or container
            if bbox.width > 200 and bbox.height > 100:
                return "container"

        # Icon-like: small and roughly square
        if 0.8 < aspect_ratio < 1.25 and bbox.width < 60 and bbox.height < 60:
            return "icon"

        # Default to container for larger elements
        if result.contour_area > 10000:
            return "container"

        return "element"

    def _create_edge_overlay(
        self,
        screenshot: np.ndarray,
        edges: np.ndarray,
        results: list[EdgeDetectionResult],
    ) -> np.ndarray:
        """
        Create debug visualization with edges and contours overlaid.

        Args:
            screenshot: Original screenshot.
            edges: Canny edge output.
            results: Filtered edge detection results.

        Returns:
            BGR image with overlays.
        """
        # Create a copy to draw on
        overlay = screenshot.copy()

        # Draw edges in cyan
        edge_mask = edges > 0
        overlay[edge_mask] = [255, 255, 0]  # Cyan in BGR

        # Draw filtered contour bounding boxes
        for result in results:
            bbox = result.bbox
            color = self._get_category_color(result)
            cv2.rectangle(
                overlay,
                (bbox.x, bbox.y),
                (bbox.x2, bbox.y2),
                color,
                2,
            )

            # Draw contour points if available
            if result.contour_points:
                points = np.array(result.contour_points, dtype=np.int32)
                cv2.polylines(overlay, [points], True, color, 1)

        return overlay

    def _get_category_color(self, result: EdgeDetectionResult) -> tuple[int, int, int]:
        """Get color for category visualization (BGR)."""
        category = self._classify_shape(result)
        colors = {
            "button": (0, 255, 0),  # Green
            "input": (255, 0, 0),  # Blue
            "icon": (0, 255, 255),  # Yellow
            "container": (128, 128, 128),  # Gray
            "element": (255, 255, 255),  # White
        }
        return colors.get(category, (255, 255, 255))

    def get_overlay_base64(self, overlay: np.ndarray) -> str:
        """
        Convert overlay image to base64 for API transport.

        Args:
            overlay: BGR numpy array.

        Returns:
            Base64-encoded PNG string.
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb)

        # Encode to PNG
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")

        # Base64 encode
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def reset_counter(self) -> None:
        """Reset the result counter (call between screenshots)."""
        self._result_counter = 0
