"""Context analyzer to determine the type of clicked element."""

import logging
from typing import Any

import cv2
import numpy as np

from .models import ElementType, InferredBoundingBox

logger = logging.getLogger(__name__)


class ClickContextAnalyzer:
    """Analyzes the context around a click to determine element type.

    Uses visual features to classify what type of GUI element was clicked:
    - Buttons: Rectangular, uniform color, possibly with text
    - Icons: Small, square-ish, often with high edge density
    - Text: Horizontal, small height, high contrast
    - Images: Large, complex, many colors
    - Checkboxes/Radios: Small squares/circles
    - Input fields: Rectangular with border
    - Links: Text-like with underline or distinct color
    - Menu items: Rectangular, part of vertical list
    - Tabs: Rectangular, part of horizontal list
    """

    def __init__(self) -> None:
        """Initialize the context analyzer."""
        pass

    def analyze_element_type(
        self,
        screenshot: np.ndarray[Any, Any],
        bbox: InferredBoundingBox,
        click_location: tuple[int, int],
    ) -> ElementType:
        """Determine the type of GUI element.

        Args:
            screenshot: Full screenshot image.
            bbox: Detected bounding box.
            click_location: Original click coordinates.

        Returns:
            Classified ElementType.
        """
        # Extract region
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height

        # Clamp to image bounds
        img_h, img_w = screenshot.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w <= 0 or h <= 0:
            return ElementType.UNKNOWN

        region = screenshot[y : y + h, x : x + w]

        # Calculate features
        features = self._extract_features(region)

        # Classify based on features
        return self._classify_element(features, w, h)

    def _extract_features(self, region: np.ndarray[Any, Any]) -> dict[str, float]:
        """Extract visual features from the region."""
        features: dict[str, float] = {}

        h, w = region.shape[:2]

        # Aspect ratio
        features["aspect_ratio"] = w / h if h > 0 else 1.0

        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        else:
            gray = region
            hsv = None

        # Color analysis
        if hsv is not None:
            # Color uniformity (low std = uniform)
            hue_std = np.std(hsv[:, :, 0])
            sat_std = np.std(hsv[:, :, 1])
            features["color_uniformity"] = 1.0 - min(1.0, (hue_std + sat_std) / 180)

            # Average saturation (buttons often have saturated colors)
            features["avg_saturation"] = float(np.mean(hsv[:, :, 1])) / 255.0

            # Number of distinct colors (approximation)
            unique_colors = len(np.unique(hsv[:, :, 0]))
            features["color_complexity"] = min(1.0, unique_colors / 50)
        else:
            features["color_uniformity"] = 0.5
            features["avg_saturation"] = 0.0
            features["color_complexity"] = 0.5

        # Edge analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features["edge_density"] = float(edge_density)

        # Border detection (edges near perimeter)
        border_width = min(3, min(w, h) // 4)
        if border_width > 0:
            border_mask = np.zeros_like(gray, dtype=bool)
            border_mask[:border_width, :] = True
            border_mask[-border_width:, :] = True
            border_mask[:, :border_width] = True
            border_mask[:, -border_width:] = True

            border_edges = np.sum(edges[border_mask] > 0)
            total_border = np.sum(border_mask)
            features["border_strength"] = (
                border_edges / total_border if total_border > 0 else 0.0
            )
        else:
            features["border_strength"] = 0.0

        # Rectangularity (how rectangular is the content)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            features["solidity"] = contour_area / hull_area if hull_area > 0 else 0.0
        else:
            features["solidity"] = 0.0

        # Brightness analysis
        features["avg_brightness"] = float(np.mean(gray)) / 255.0
        features["brightness_std"] = float(np.std(gray)) / 128.0

        # Check for text-like horizontal structure
        horizontal_projection = np.sum(gray, axis=1)
        proj_variance = np.var(horizontal_projection)
        features["horizontal_structure"] = min(1.0, proj_variance / (w * 255 * 10))

        # Circularity for icon/checkbox detection
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter**2)
                features["circularity"] = float(circularity)
            else:
                features["circularity"] = 0.0
        else:
            features["circularity"] = 0.0

        return features

    def _classify_element(
        self,
        features: dict[str, float],
        width: int,
        height: int,
    ) -> ElementType:
        """Classify element based on extracted features."""
        aspect = features.get("aspect_ratio", 1.0)
        edge_density = features.get("edge_density", 0.0)
        color_uniformity = features.get("color_uniformity", 0.5)
        border_strength = features.get("border_strength", 0.0)
        circularity = features.get("circularity", 0.0)
        saturation = features.get("avg_saturation", 0.0)
        color_complexity = features.get("color_complexity", 0.5)
        horizontal_structure = features.get("horizontal_structure", 0.0)

        area = width * height

        # Score each element type
        scores: dict[ElementType, float] = {}

        # CHECKBOX: Small, square, often has checkmark pattern
        if 10 <= width <= 30 and 10 <= height <= 30:
            score = 0.0
            if 0.8 <= aspect <= 1.2:  # Square
                score += 0.4
            if border_strength > 0.3:
                score += 0.3
            if edge_density > 0.1:
                score += 0.2
            scores[ElementType.CHECKBOX] = score

        # RADIO: Small, circular
        if 10 <= width <= 30 and 10 <= height <= 30:
            score = 0.0
            if 0.8 <= aspect <= 1.2 and circularity > 0.7:
                score += 0.6
            if border_strength > 0.2:
                score += 0.2
            scores[ElementType.RADIO] = score

        # ICON: Small to medium, roughly square, moderate edge complexity
        if 16 <= width <= 80 and 16 <= height <= 80:
            score = 0.0
            if 0.7 <= aspect <= 1.4:  # Roughly square
                score += 0.3
            if 0.15 <= edge_density <= 0.5:  # Moderate edge complexity
                score += 0.3
            if color_complexity < 0.5:  # Icons often have few colors
                score += 0.2
            scores[ElementType.ICON] = score

        # BUTTON: Wider than tall, uniform color, possibly bordered
        score = 0.0
        if 1.5 <= aspect <= 8.0:  # Wider than tall
            score += 0.25
        if color_uniformity > 0.6:
            score += 0.25
        if 40 <= width <= 300 and 20 <= height <= 60:
            score += 0.25
        if saturation > 0.2:  # Buttons often have colored backgrounds
            score += 0.15
        if border_strength > 0.2:
            score += 0.1
        scores[ElementType.BUTTON] = score

        # TEXT: Horizontal, small height, high horizontal structure
        score = 0.0
        if aspect > 2.0:  # Much wider than tall
            score += 0.3
        if 10 <= height <= 30:  # Text-like height
            score += 0.3
        if horizontal_structure > 0.3:
            score += 0.3
        if edge_density > 0.1:
            score += 0.1
        scores[ElementType.TEXT] = score

        # LINK: Similar to text but may have underline
        score = scores.get(ElementType.TEXT, 0.0) * 0.8
        if saturation > 0.3:  # Links often blue/colored
            score += 0.2
        scores[ElementType.LINK] = score

        # INPUT_FIELD: Rectangular, bordered, often light background
        score = 0.0
        if 2.0 <= aspect <= 10.0:
            score += 0.2
        if border_strength > 0.4:
            score += 0.3
        if color_uniformity > 0.7:
            score += 0.2
        if features.get("avg_brightness", 0) > 0.7:  # Light background
            score += 0.2
        if 100 <= width <= 400 and 20 <= height <= 50:
            score += 0.1
        scores[ElementType.INPUT_FIELD] = score

        # IMAGE: Large, complex colors, varied content
        score = 0.0
        if area > 10000:  # Large area
            score += 0.3
        if color_complexity > 0.5:
            score += 0.3
        if features.get("brightness_std", 0) > 0.3:  # Varied brightness
            score += 0.2
        scores[ElementType.IMAGE] = score

        # MENU_ITEM: Rectangular, part of vertical list (hard to detect without context)
        score = 0.0
        if 2.0 <= aspect <= 8.0:
            score += 0.2
        if 80 <= width <= 300 and 20 <= height <= 40:
            score += 0.3
        if color_uniformity > 0.5:
            score += 0.2
        scores[ElementType.MENU_ITEM] = score

        # TAB: Similar to button but typically in a horizontal group
        score = scores.get(ElementType.BUTTON, 0.0) * 0.7
        if 60 <= width <= 150:
            score += 0.2
        scores[ElementType.TAB] = score

        # Find highest scoring type
        if not scores:
            return ElementType.UNKNOWN

        best_type = max(scores, key=lambda t: scores[t])
        best_score = scores[best_type]

        # Require minimum confidence
        if best_score < 0.3:
            return ElementType.UNKNOWN

        return best_type

    def get_element_type_confidence(
        self,
        screenshot: np.ndarray[Any, Any],
        bbox: InferredBoundingBox,
        click_location: tuple[int, int],
    ) -> tuple[ElementType, float]:
        """Get element type with confidence score.

        Args:
            screenshot: Full screenshot image.
            bbox: Detected bounding box.
            click_location: Original click coordinates.

        Returns:
            Tuple of (ElementType, confidence).
        """
        element_type = self.analyze_element_type(screenshot, bbox, click_location)

        # Calculate confidence based on how well features match the type
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
        img_h, img_w = screenshot.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w <= 0 or h <= 0:
            return ElementType.UNKNOWN, 0.0

        region = screenshot[y : y + h, x : x + w]
        features = self._extract_features(region)

        # Simple confidence based on feature clarity
        confidence = 0.5

        if element_type == ElementType.BUTTON:
            if features.get("color_uniformity", 0) > 0.7:
                confidence += 0.2
            if 1.5 <= features.get("aspect_ratio", 0) <= 6.0:
                confidence += 0.2

        elif element_type == ElementType.ICON:
            if 0.7 <= features.get("aspect_ratio", 0) <= 1.4:
                confidence += 0.2
            if 0.15 <= features.get("edge_density", 0) <= 0.5:
                confidence += 0.2

        elif element_type in (ElementType.CHECKBOX, ElementType.RADIO):
            if 0.8 <= features.get("aspect_ratio", 0) <= 1.2:
                confidence += 0.3

        elif element_type == ElementType.TEXT:
            if features.get("aspect_ratio", 0) > 2.0:
                confidence += 0.2
            if features.get("horizontal_structure", 0) > 0.3:
                confidence += 0.2

        return element_type, min(0.95, confidence)
