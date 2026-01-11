"""Element Pattern Detector for GUI Environment Discovery.

Detects and classifies UI element patterns including:
- Buttons, input fields, cards
- Size ranges and typical colors
- Shape analysis (rectangle, rounded, circular)
- Shadow and border detection
"""

import hashlib
import logging
from collections import defaultdict
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.environment import (
    BoundingBox,
    ElementPattern,
    ElementPatterns,
    ElementSample,
    ElementShape,
    SizeRange,
)

from qontinui.vision.environment.analyzers.base import BaseAnalyzer

logger = logging.getLogger(__name__)


class ElementPatternDetector(BaseAnalyzer[ElementPatterns]):
    """Detects and classifies UI element patterns from screenshots.

    Uses object detection or heuristics to identify common UI elements
    and learn their visual patterns.
    """

    def __init__(
        self,
        min_element_size: int = 20,
        max_element_size: int = 800,
        use_ml_detection: bool = True,
    ) -> None:
        """Initialize the element pattern detector.

        Args:
            min_element_size: Minimum element dimension (pixels).
            max_element_size: Maximum element dimension (pixels).
            use_ml_detection: Whether to use ML-based detection if available.
        """
        super().__init__("ElementPatternDetector")
        self.min_element_size = min_element_size
        self.max_element_size = max_element_size
        self.use_ml_detection = use_ml_detection
        self._detector = None

    async def analyze(
        self,
        screenshots: list[NDArray[np.uint8]],
        element_annotations: list[list[dict[str, Any]]] | None = None,
        ocr_results: list[list[dict[str, Any]]] | None = None,
        **kwargs: Any,
    ) -> ElementPatterns:
        """Analyze screenshots to detect element patterns.

        Args:
            screenshots: List of screenshots as numpy arrays (BGR format).
            element_annotations: Optional pre-annotated elements.
            ocr_results: Optional OCR results for text style analysis.

        Returns:
            ElementPatterns with detected patterns.
        """
        self.reset()

        if not screenshots:
            return ElementPatterns(confidence=0.0)

        self._log_progress(f"Analyzing {len(screenshots)} screenshots")

        # Detect elements in screenshots
        all_elements: list[dict[str, Any]] = []

        if element_annotations:
            # Use provided annotations
            for annotations in element_annotations:
                all_elements.extend(annotations)
        else:
            # Auto-detect elements
            for idx, screenshot in enumerate(screenshots):
                screenshot = self._ensure_bgr(screenshot)
                elements = await self._detect_elements(screenshot, idx)
                all_elements.extend(elements)

        if not all_elements:
            return ElementPatterns(
                screenshots_analyzed=len(screenshots),
                confidence=0.0,
            )

        # Cluster elements by type
        element_clusters = self._cluster_elements(all_elements)

        # Create patterns from clusters
        patterns: dict[str, ElementPattern] = {}
        for element_type, elements in element_clusters.items():
            pattern = self._create_pattern(element_type, elements, screenshots)
            if pattern:
                patterns[element_type] = pattern

        self._screenshots_analyzed = len(screenshots)
        self.confidence = self._calculate_confidence(
            len(all_elements),
            min_samples=5,
            optimal_samples=50,
        )

        return ElementPatterns(
            patterns=patterns,
            screenshots_analyzed=len(screenshots),
            elements_detected=len(all_elements),
            confidence=self.confidence,
        )

    async def _detect_elements(
        self,
        screenshot: NDArray[np.uint8],
        screenshot_idx: int,
    ) -> list[dict[str, Any]]:
        """Detect UI elements in a screenshot.

        Args:
            screenshot: BGR screenshot.
            screenshot_idx: Index of screenshot.

        Returns:
            List of detected element dictionaries.
        """
        elements = []

        # Try ML-based detection first
        if self.use_ml_detection:
            ml_elements = await self._ml_detect_elements(screenshot)
            if ml_elements:
                for elem in ml_elements:
                    elem["screenshot_idx"] = screenshot_idx
                return ml_elements

        # Fallback to heuristic detection
        elements = self._heuristic_detect_elements(screenshot)
        for elem in elements:
            elem["screenshot_idx"] = screenshot_idx

        return elements

    async def _ml_detect_elements(
        self,
        screenshot: NDArray[np.uint8],
    ) -> list[dict[str, Any]] | None:
        """Detect elements using ML model (YOLO, etc.).

        Args:
            screenshot: BGR screenshot.

        Returns:
            List of detected elements or None if unavailable.
        """
        # TODO: Use ultralytics YOLO if available with UI-specific model
        # For now, return None to use heuristics
        if self._detector is None:
            return None

        # Would need a fine-tuned model for UI elements
        # results = self._detector(screenshot)
        # ...process results...

        return None

    def _heuristic_detect_elements(
        self,
        screenshot: NDArray[np.uint8],
    ) -> list[dict[str, Any]]:
        """Detect elements using heuristics and edge detection.

        Args:
            screenshot: BGR screenshot.

        Returns:
            List of detected element dictionaries.
        """
        elements = []

        try:
            import cv2

            h, w = screenshot.shape[:2]

            # Convert to grayscale
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

            # Edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Get bounding rectangle
                x, y, cw, ch = cv2.boundingRect(contour)

                # Filter by size
                if cw < self.min_element_size or ch < self.min_element_size:
                    continue
                if cw > self.max_element_size or ch > self.max_element_size:
                    continue

                # Skip very elongated elements (likely separators)
                aspect_ratio = cw / ch if ch > 0 else 0
                if aspect_ratio > 15 or aspect_ratio < 0.067:
                    continue

                # Extract element region
                region = screenshot[y : y + ch, x : x + cw]

                # Classify element type
                element_type = self._classify_element(region, cw, ch, aspect_ratio)

                # Detect shape - cast contour to expected type for mypy
                shape = self._detect_shape(
                    cast(NDArray[np.int32], contour), region
                )

                # Get dominant color
                dominant_color = self._get_dominant_color(region)

                elements.append(
                    {
                        "bbox": (x, y, cw, ch),
                        "type": element_type,
                        "shape": shape,
                        "color": dominant_color,
                        "aspect_ratio": aspect_ratio,
                        "area": cw * ch,
                    }
                )

        except ImportError:
            # Simple fallback
            elements = self._simple_element_detection(screenshot)

        return elements

    def _simple_element_detection(
        self,
        screenshot: NDArray[np.uint8],
    ) -> list[dict[str, Any]]:
        """Simple element detection without OpenCV.

        Args:
            screenshot: BGR screenshot.

        Returns:
            Basic element list.
        """
        # Very simple: detect uniform color regions
        h, w = screenshot.shape[:2]
        elements = []

        # Quantize colors
        quantized = (screenshot // 64) * 64
        flat = quantized.reshape(-1, 3)

        # Find unique colors
        unique = np.unique(flat, axis=0)

        for color in unique:
            mask = np.all(quantized == color, axis=2)
            if mask.sum() < self.min_element_size * self.min_element_size:
                continue

            # Find bounds of this color region
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                continue

            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            cw = x_max - x_min
            ch = y_max - y_min

            if cw >= self.min_element_size and ch >= self.min_element_size:
                b, g, r = color
                elements.append(
                    {
                        "bbox": (x_min, y_min, cw, ch),
                        "type": "unknown",
                        "shape": ElementShape.RECTANGLE,
                        "color": self._rgb_to_hex(int(r), int(g), int(b)),
                        "aspect_ratio": cw / ch if ch > 0 else 1,
                        "area": cw * ch,
                    }
                )

        return elements

    def _classify_element(
        self,
        region: NDArray[np.uint8],
        width: int,
        height: int,
        aspect_ratio: float,
    ) -> str:
        """Classify element type based on characteristics.

        Args:
            region: Element image region.
            width: Element width.
            height: Element height.
            aspect_ratio: Width/height ratio.

        Returns:
            Element type string.
        """
        # Button: typically wider than tall, moderate size
        if 2 < aspect_ratio < 8 and 25 < height < 60 and 60 < width < 300:
            return "button"

        # Input field: very wide, specific height range
        if aspect_ratio > 4 and 30 < height < 50:
            return "input_field"

        # Card: larger, roughly square-ish
        if 0.5 < aspect_ratio < 2 and width > 150 and height > 100:
            return "card"

        # Icon: small and square
        if 0.8 < aspect_ratio < 1.2 and width < 50 and height < 50:
            return "icon"

        # List item: wide and short
        if aspect_ratio > 5 and 30 < height < 80:
            return "list_item"

        # Checkbox/toggle: small and square
        if 0.8 < aspect_ratio < 1.2 and 15 < width < 40:
            return "checkbox"

        return "unknown"

    def _detect_shape(
        self,
        contour: NDArray[np.int32],
        region: NDArray[np.uint8],
    ) -> ElementShape:
        """Detect element shape from contour.

        Args:
            contour: OpenCV contour.
            region: Element image region.

        Returns:
            ElementShape classification.
        """
        try:
            import cv2

            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check circularity
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.8:
                    # Check if roughly square (circle) vs elongated (oval)
                    x, y, w, h = cv2.boundingRect(contour)
                    if 0.8 < w / h < 1.2:
                        return ElementShape.CIRCLE
                    return ElementShape.OVAL

            # Check for rectangle
            if len(approx) == 4:
                # Check if corners are sharp or rounded
                corner_radius = self._estimate_corner_radius(region)
                if corner_radius > 3:
                    return ElementShape.ROUNDED_RECTANGLE
                return ElementShape.RECTANGLE

            return ElementShape.IRREGULAR

        except Exception:
            return ElementShape.RECTANGLE

    def _estimate_corner_radius(
        self,
        region: NDArray[np.uint8],
    ) -> int:
        """Estimate corner radius from element region.

        Args:
            region: Element image region.

        Returns:
            Estimated corner radius in pixels.
        """
        try:
            import cv2

            h, w = region.shape[:2]
            corner_size = min(h, w) // 4

            # Sample corners
            corners = [
                region[:corner_size, :corner_size],
                region[:corner_size, -corner_size:],
                region[-corner_size:, :corner_size],
                region[-corner_size:, -corner_size:],
            ]

            # Check for rounded corners by looking at diagonal pixels
            total_corner_pixels = 0
            empty_corner_pixels = 0

            for corner in corners:
                gray = (
                    cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY) if len(corner.shape) == 3 else corner
                )
                # Check if corners have empty (background) pixels in diagonal
                ch, cw = gray.shape
                for i in range(min(ch, cw) // 2):
                    if gray[i, i] < 30 or gray[i, cw - 1 - i] < 30:
                        empty_corner_pixels += 1
                    total_corner_pixels += 2

            if total_corner_pixels > 0:
                empty_ratio = empty_corner_pixels / total_corner_pixels
                # Estimate radius from empty ratio
                return int(corner_size * empty_ratio)

        except Exception:
            pass

        return 0

    def _get_dominant_color(
        self,
        region: NDArray[np.uint8],
    ) -> str:
        """Get dominant color of element.

        Args:
            region: Element image region.

        Returns:
            Hex color string.
        """
        # Simple average color
        flat = region.reshape(-1, 3)
        avg = flat.mean(axis=0).astype(int)
        b, g, r = avg
        return self._rgb_to_hex(int(r), int(g), int(b))

    def _cluster_elements(
        self,
        elements: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Cluster elements by type.

        Args:
            elements: All detected elements.

        Returns:
            Dictionary mapping element type to list of elements.
        """
        clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for element in elements:
            element_type = element.get("type", "unknown")
            clusters[element_type].append(element)

        return dict(clusters)

    def _create_pattern(
        self,
        element_type: str,
        elements: list[dict[str, Any]],
        screenshots: list[NDArray[np.uint8]],
    ) -> ElementPattern | None:
        """Create a pattern from a cluster of similar elements.

        Args:
            element_type: Type of element.
            elements: List of similar elements.
            screenshots: Original screenshots for sample extraction.

        Returns:
            ElementPattern or None if insufficient data.
        """
        if len(elements) < 2:
            return None

        # Calculate size ranges
        widths = [e["bbox"][2] for e in elements]
        heights = [e["bbox"][3] for e in elements]

        width_range = SizeRange(min=min(widths), max=max(widths))
        height_range = SizeRange(min=min(heights), max=max(heights))

        # Get typical colors
        colors: list[str] = [
            c for e in elements if (c := e.get("color")) is not None and isinstance(c, str)
        ]
        color_counts: dict[str, int] = defaultdict(int)
        for c in colors:
            color_counts[c] += 1
        typical_colors = [c for c, _ in sorted(color_counts.items(), key=lambda x: -x[1])[:5]]

        # Get most common shape
        shapes = [e.get("shape", ElementShape.RECTANGLE) for e in elements]
        shape_counts: dict[ElementShape, int] = defaultdict(int)
        for s in shapes:
            shape_counts[s] += 1
        most_common_shape = max(shape_counts.keys(), key=lambda s: shape_counts[s])

        # Estimate corner radius
        corner_radii = [e.get("corner_radius", 0) for e in elements if e.get("corner_radius")]
        avg_corner_radius = int(sum(corner_radii) / len(corner_radii)) if corner_radii else None

        # Detect shadow (simplified)
        has_shadow = self._detect_shadow_pattern(elements, screenshots)

        # Create sample images
        samples = self._create_samples(elements, screenshots)

        # Confidence based on consistency
        size_consistency = 1 - np.std(widths) / (np.mean(widths) + 1)
        confidence = min(1.0, size_consistency * len(elements) / 10)

        return ElementPattern(
            element_type=element_type,
            typical_width=width_range,
            typical_height=height_range,
            typical_colors=typical_colors,
            shape=most_common_shape,
            corner_radius=avg_corner_radius,
            has_shadow=has_shadow,
            examples=samples,
            detection_count=len(elements),
            confidence=confidence,
        )

    def _detect_shadow_pattern(
        self,
        elements: list[dict[str, Any]],
        screenshots: list[NDArray[np.uint8]],
    ) -> bool:
        """Detect if elements typically have shadows.

        Args:
            elements: Element list.
            screenshots: Screenshots for analysis.

        Returns:
            True if shadows detected.
        """
        # Check a sample of elements for shadow
        shadow_count = 0
        checked = 0

        for element in elements[:5]:
            screenshot_idx = element.get("screenshot_idx", 0)
            if screenshot_idx >= len(screenshots):
                continue

            screenshot = screenshots[screenshot_idx]
            x, y, w, h = element["bbox"]

            # Check area below element for darker pixels (shadow)
            shadow_region_y = min(y + h + 2, screenshot.shape[0] - 5)
            shadow_region = screenshot[shadow_region_y : shadow_region_y + 5, x : x + w]

            if shadow_region.size > 0:
                # Compare brightness with element
                element_region = screenshot[y : y + h, x : x + w]
                if element_region.size > 0:
                    element_brightness = element_region.mean()
                    shadow_brightness = shadow_region.mean()

                    if shadow_brightness < element_brightness * 0.8:
                        shadow_count += 1
                    checked += 1

        return checked > 0 and shadow_count > checked / 2

    def _create_samples(
        self,
        elements: list[dict[str, Any]],
        screenshots: list[NDArray[np.uint8]],
    ) -> list[ElementSample]:
        """Create sample images for the pattern.

        Args:
            elements: Elements to sample.
            screenshots: Source screenshots.

        Returns:
            List of ElementSample objects.
        """
        samples = []

        for element in elements[:5]:  # Max 5 samples
            screenshot_idx = element.get("screenshot_idx", 0)
            if screenshot_idx >= len(screenshots):
                continue

            x, y, w, h = element["bbox"]

            # Create perceptual hash (simplified)
            screenshot = screenshots[screenshot_idx]
            region = screenshot[y : y + h, x : x + w]
            image_hash = self._compute_image_hash(region)

            samples.append(
                ElementSample(
                    image_hash=image_hash,
                    bounds=BoundingBox(x=x, y=y, width=w, height=h),
                )
            )

        return samples

    def _compute_image_hash(
        self,
        image: NDArray[np.uint8],
    ) -> str:
        """Compute a simple perceptual hash for an image.

        Args:
            image: Image region.

        Returns:
            Hash string.
        """
        try:
            import cv2

            # Resize to 8x8
            small = cv2.resize(image, (8, 8))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small

            # Compute average
            avg = gray.mean()

            # Create hash from comparison to average
            bits = (gray > avg).flatten()
            hash_bytes = np.packbits(bits)

            return hash_bytes.tobytes().hex()

        except ImportError:
            # Fallback: use MD5 of downsampled pixels
            small = image[::8, ::8] if image.size > 64 else image
            return hashlib.md5(small.tobytes(), usedforsecurity=False).hexdigest()[:16]
