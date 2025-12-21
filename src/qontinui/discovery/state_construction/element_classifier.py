"""Element type classification for UI analysis.

This module provides specialized element classification capabilities using size,
shape, visual features, and position to determine element types like buttons,
icons, text fields, and window controls.
"""

from typing import Any

import cv2
import numpy as np

from qontinui.discovery.models import StateImage
from qontinui.discovery.state_construction.element_identifier import ElementType


class ElementClassifier:
    """Classifies UI element types based on visual and contextual features.

    Uses size, shape, aspect ratio, edge density, color analysis, and
    position to determine the likely element type (button, icon, text
    field, checkbox, etc.).
    """

    def __init__(
        self,
        button_aspect_ratio_range: tuple[float, float] = (0.3, 4.0),
        icon_size_range: tuple[int, int] = (16, 128),
        title_bar_height_range: tuple[int, int] = (20, 60),
    ):
        """Initialize the element classifier.

        Args:
            button_aspect_ratio_range: (min, max) aspect ratio for buttons
            icon_size_range: (min, max) size for icons
            title_bar_height_range: (min, max) height for title bars
        """
        self.button_aspect_ratio_range = button_aspect_ratio_range
        self.icon_size_range = icon_size_range
        self.title_bar_height_range = title_bar_height_range

    def classify_image_type(
        self, state_image: StateImage, screenshot: np.ndarray | None = None
    ) -> ElementType:
        """Classify the type of a StateImage element.

        Uses size, shape, visual features, and position to determine
        the likely element type (button, icon, text field, etc.).

        Args:
            state_image: StateImage to classify
            screenshot: Optional full screenshot for context

        Returns:
            Classified element type
        """
        width = state_image.width
        height = state_image.height
        aspect_ratio = width / height if height > 0 else 0

        # Extract properties for classification
        properties = self._extract_element_properties(state_image, screenshot)

        # Window controls (small, square/round, in top corner)
        if self._is_window_control(state_image, properties):
            return ElementType.WINDOW_CONTROL

        # Icons (small, roughly square)
        if width <= self.icon_size_range[1] and height <= self.icon_size_range[1]:
            if 0.7 <= aspect_ratio <= 1.3:
                return ElementType.ICON

        # Buttons (medium size, reasonable aspect ratio)
        if self._is_button_like(state_image, properties, aspect_ratio):
            return ElementType.BUTTON

        # Text fields (elongated horizontally, specific characteristics)
        if aspect_ratio > 2.5 and self._has_text_field_characteristics(properties):
            return ElementType.TEXT_FIELD

        # Dividers (very thin in one dimension)
        if width < 5 or height < 5:
            return ElementType.DIVIDER

        # Text (based on OCR or text-like patterns)
        if self._appears_to_be_text(properties):
            return ElementType.TEXT

        # Checkboxes and radio buttons (small, square)
        if 10 <= width <= 30 and 10 <= height <= 30 and 0.8 <= aspect_ratio <= 1.2:
            if self._has_checkbox_characteristics(properties):
                return ElementType.CHECKBOX

        # Logo (larger, often in specific positions)
        if self._is_likely_logo(state_image, properties):
            return ElementType.LOGO

        return ElementType.UNKNOWN

    def _extract_element_properties(
        self, state_image: StateImage, screenshot: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Extract visual and contextual properties of an element.

        Args:
            state_image: StateImage to analyze
            screenshot: Optional full screenshot for context

        Returns:
            Dictionary of element properties
        """
        properties = {
            "width": state_image.width,
            "height": state_image.height,
            "aspect_ratio": (
                state_image.width / state_image.height if state_image.height > 0 else 0
            ),
            "area": state_image.width * state_image.height,
            "position": (state_image.x, state_image.y),
            "frequency": state_image.frequency,
            "mask_density": state_image.mask_density,
        }

        # Add pixel data properties if available
        if state_image.pixel_data is not None:
            properties.update(self._analyze_pixel_data(state_image.pixel_data))

        # Add position-based properties
        if screenshot is not None:
            screen_height, screen_width = screenshot.shape[:2]
            properties["relative_x"] = state_image.x / screen_width
            properties["relative_y"] = state_image.y / screen_height
            properties["in_top_region"] = state_image.y < screen_height * 0.15
            properties["in_corner"] = self._is_in_corner(state_image, screen_width, screen_height)

        return properties

    def _analyze_pixel_data(self, pixel_data: np.ndarray) -> dict[str, Any]:
        """Analyze pixel data to extract visual features.

        Args:
            pixel_data: Pixel data array (BGR or grayscale)

        Returns:
            Dictionary of visual features (colors, edges, texture)
        """
        features = {}

        # Color analysis
        if len(pixel_data.shape) == 3:
            mean_color = np.mean(pixel_data, axis=(0, 1))
            std_color = np.std(pixel_data, axis=(0, 1))
            features["mean_color"] = mean_color.tolist()
            features["std_color"] = std_color.tolist()
            features["color_variance"] = np.mean(std_color)

            # Convert to grayscale for edge analysis
            gray = cv2.cvtColor(pixel_data, cv2.COLOR_BGR2GRAY)
        else:
            gray = pixel_data
            features["color_variance"] = np.std(gray)

        # Edge density (indicates complexity)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features["edge_density"] = float(edge_density)

        # Texture analysis (using local standard deviation)
        kernel_size = 5
        if gray.shape[0] > kernel_size and gray.shape[1] > kernel_size:
            blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            texture = np.std(gray - blurred)
            features["texture"] = float(texture)
        else:
            features["texture"] = 0.0

        return features

    def _is_window_control(self, state_image: StateImage, properties: dict[str, Any]) -> bool:
        """Check if element appears to be a window control button.

        Args:
            state_image: StateImage to check
            properties: Extracted properties

        Returns:
            True if likely a window control
        """
        # Small size
        if not (10 <= state_image.width <= 40 and 10 <= state_image.height <= 40):
            return False

        # In top corner or top-right area
        if not properties.get("in_top_region", False):
            return False

        # Roughly square or circular
        aspect_ratio = properties["aspect_ratio"]
        if not (0.7 <= aspect_ratio <= 1.3):
            return False

        # High frequency (appears consistently)
        if state_image.frequency < 0.7:
            return False

        return True

    def _is_button_like(
        self, state_image: StateImage, properties: dict[str, Any], aspect_ratio: float
    ) -> bool:
        """Check if element has button-like characteristics.

        Args:
            state_image: StateImage to check
            properties: Extracted properties
            aspect_ratio: Width/height ratio

        Returns:
            True if likely a button
        """
        # Reasonable size for a button
        if state_image.width < 30 or state_image.height < 15:
            return False

        # Button-like aspect ratio
        if not (
            self.button_aspect_ratio_range[0] <= aspect_ratio <= self.button_aspect_ratio_range[1]
        ):
            return False

        # Has some texture/content
        if properties.get("edge_density", 0) < 0.01:
            return False

        return True

    def _has_text_field_characteristics(self, properties: dict[str, Any]) -> bool:
        """Check if element has text field characteristics.

        Args:
            properties: Element properties

        Returns:
            True if likely a text field
        """
        # Text fields typically have low edge density (empty or simple text)
        edge_density = properties.get("edge_density", 1.0)
        if edge_density > 0.3:
            return False

        # Usually have a border or distinct background
        color_variance = properties.get("color_variance", 0)
        if color_variance > 50:  # Too much variation
            return False

        return True

    def _appears_to_be_text(self, properties: dict[str, Any]) -> bool:
        """Check if element appears to contain text.

        Args:
            properties: Element properties

        Returns:
            True if likely text
        """
        # Text has moderate edge density
        edge_density = properties.get("edge_density", 0)
        if not (0.05 <= edge_density <= 0.4):
            return False

        # Text usually has some texture
        texture = properties.get("texture", 0)
        if texture < 5:
            return False

        return True

    def _has_checkbox_characteristics(self, properties: dict[str, Any]) -> bool:
        """Check if element has checkbox characteristics.

        Args:
            properties: Element properties

        Returns:
            True if likely a checkbox
        """
        # Checkboxes have simple geometry
        edge_density = properties.get("edge_density", 0)

        # Should have clear edges (border) but not too complex
        return 0.1 <= edge_density <= 0.5  # type: ignore[no-any-return]

    def _is_likely_logo(self, state_image: StateImage, properties: dict[str, Any]) -> bool:
        """Check if element is likely a logo.

        Args:
            state_image: StateImage to check
            properties: Element properties

        Returns:
            True if likely a logo
        """
        # Logos are often in top-left or have high visibility
        if not properties.get("in_top_region", False):
            return False

        # Reasonable logo size
        if state_image.width < 50 or state_image.height < 20:
            return False

        # High frequency (consistent across screenshots)
        if state_image.frequency < 0.8:
            return False

        # Often has higher visual complexity
        edge_density = properties.get("edge_density", 0)
        if edge_density < 0.1:
            return False

        return True

    def _is_in_corner(self, state_image: StateImage, screen_width: int, screen_height: int) -> bool:
        """Check if element is in a screen corner.

        Args:
            state_image: StateImage to check
            screen_width: Screen width
            screen_height: Screen height

        Returns:
            True if in corner
        """
        corner_threshold = 0.1  # 10% from edges

        x_left = state_image.x < screen_width * corner_threshold
        x_right = state_image.x2 > screen_width * (1 - corner_threshold)
        y_top = state_image.y < screen_height * corner_threshold
        y_bottom = state_image.y2 > screen_height * (1 - corner_threshold)

        return (x_left or x_right) and (y_top or y_bottom)
