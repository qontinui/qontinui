"""Basic description generator using visual features."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .base import DescriptionGenerator


class BasicDescriptionGenerator(DescriptionGenerator):
    """Generate basic descriptions using visual features.

    This is a fallback generator that doesn't require ML models.
    It analyzes basic visual properties like color, shape, size, and position.
    """

    def __init__(self) -> None:
        """Initialize basic generator."""
        super().__init__()

    def generate(
        self,
        image: np.ndarray[Any, Any],
        mask: np.ndarray[Any, Any] | None = None,
        bbox: tuple[Any, ...] | None = None,
    ) -> str:
        """Generate description from visual features.

        Args:
            image: Input image (BGR format)
            mask: Optional binary mask
            bbox: Optional bounding box (x, y, w, h)

        Returns:
            Basic description based on visual features
        """
        # Preprocess
        region = self.preprocess_image(image, mask, bbox)

        properties = []

        # Analyze size
        size_desc = self._describe_size(region, image)
        if size_desc:
            properties.append(size_desc)

        # Analyze shape
        shape_desc = self._describe_shape(region, mask)
        if shape_desc:
            properties.append(shape_desc)

        # Analyze color
        color_desc = self._describe_color(region, mask)
        if color_desc:
            properties.append(color_desc)

        # Analyze position
        if bbox:
            pos_desc = self._describe_position(bbox, image.shape)
            if pos_desc:
                properties.extend(pos_desc)

        # Analyze texture/patterns
        texture_desc = self._describe_texture(region)
        if texture_desc:
            properties.append(texture_desc)

        # Combine properties
        if properties:
            return " ".join(properties) + " element"
        else:
            return "unidentified element"

    def batch_generate(
        self, image: np.ndarray[Any, Any], regions: list[Any]
    ) -> list[str]:
        """Generate descriptions for multiple regions.

        Args:
            image: Full image
            regions: List of dicts with 'mask' and/or 'bbox'

        Returns:
            List of descriptions
        """
        descriptions = []
        for region_data in regions:
            mask = region_data.get("mask")
            bbox = region_data.get("bbox")
            desc = self.generate(image, mask, bbox)
            descriptions.append(desc)
        return descriptions

    def _describe_size(
        self, region: np.ndarray[Any, Any], full_image: np.ndarray[Any, Any]
    ) -> str | None:
        """Describe size relative to full image.

        Args:
            region: Region to analyze
            full_image: Full image for comparison

        Returns:
            Size description or None
        """
        if region.size == 0 or full_image.size == 0:
            return None

        region_area = region.shape[0] * region.shape[1]
        full_area = full_image.shape[0] * full_image.shape[1]
        ratio = region_area / full_area

        if ratio > 0.5:
            return "large"
        elif ratio > 0.1:
            return "medium-sized"
        elif ratio > 0.01:
            return "small"
        else:
            return "tiny"

    def _describe_shape(
        self, region: np.ndarray[Any, Any], mask: np.ndarray[Any, Any] | None
    ) -> str | None:
        """Describe shape characteristics.

        Args:
            region: Region to analyze
            mask: Optional mask for shape analysis

        Returns:
            Shape description or None
        """
        if region.shape[0] == 0 or region.shape[1] == 0:
            return None

        h, w = region.shape[:2]
        aspect_ratio = w / h

        # Basic shape from aspect ratio
        if 0.9 <= aspect_ratio <= 1.1:
            shape = "square"
        elif aspect_ratio > 3:
            shape = "horizontal bar"
        elif aspect_ratio < 0.33:
            shape = "vertical bar"
        elif aspect_ratio > 1.5:
            shape = "wide rectangular"
        elif aspect_ratio < 0.67:
            shape = "tall rectangular"
        else:
            shape = "rectangular"

        # If mask provided, analyze for roundness
        if mask is not None:
            roundness = self._calculate_roundness(mask)
            if roundness > 0.8:
                shape = "circular"
            elif roundness > 0.6 and "square" in shape:
                shape = "rounded square"

        return shape

    def _calculate_roundness(self, mask: np.ndarray[Any, Any]) -> float:
        """Calculate how round a shape is.

        Args:
            mask: Binary mask

        Returns:
            Roundness score (0 to 1, where 1 is perfectly circular)
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return 0.0

        # Use largest contour
        contour = max(contours, key=cv2.contourArea)

        # Calculate roundness using area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            return 0.0

        roundness = 4 * np.pi * area / (perimeter * perimeter)
        return min(1.0, roundness)

    def _describe_color(
        self, region: np.ndarray[Any, Any], mask: np.ndarray[Any, Any] | None
    ) -> str | None:
        """Describe dominant colors.

        Args:
            region: Region to analyze (BGR format)
            mask: Optional mask to focus on specific pixels

        Returns:
            Color description or None
        """
        if region.size == 0 or len(region.shape) < 3:
            return None

        # Apply mask if provided
        if mask is not None:
            pixels = region[mask > 0]
        else:
            pixels = region.reshape(-1, region.shape[-1])

        if len(pixels) == 0:
            return None

        # Calculate mean color
        mean_bgr = np.mean(pixels, axis=0)
        b, g, r = mean_bgr[:3]

        # Determine dominant color
        colors = []

        # Check for grayscale
        if abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20:
            if r > 200:
                colors.append("white")
            elif r < 50:
                colors.append("black")
            else:
                colors.append("gray")
        else:
            # Determine dominant channel
            if r > g and r > b:
                if r - max(g, b) > 50:
                    colors.append("red")
                elif g > b:
                    colors.append("orange")
                else:
                    colors.append("pink")
            elif g > r and g > b:
                if g - max(r, b) > 50:
                    colors.append("green")
                elif b > r:
                    colors.append("teal")
                else:
                    colors.append("yellow-green")
            elif b > r and b > g:
                if b - max(r, g) > 50:
                    colors.append("blue")
                elif g > r:
                    colors.append("cyan")
                else:
                    colors.append("purple")

        # Check brightness
        brightness = (r + g + b) / 3
        if brightness > 200 and len(colors) > 0:
            colors[0] = "light " + colors[0]
        elif brightness < 80 and len(colors) > 0:
            colors[0] = "dark " + colors[0]

        return colors[0] if colors else "colored"

    def _describe_position(
        self, bbox: tuple[Any, ...], image_shape: tuple[Any, ...]
    ) -> list[str]:
        """Describe position in image.

        Args:
            bbox: Bounding box (x, y, w, h)
            image_shape: Shape of full image

        Returns:
            List of position descriptions
        """
        x, y, w, h = bbox
        img_h, img_w = image_shape[:2]

        positions = []

        # Vertical position
        center_y = y + h / 2
        if center_y < img_h * 0.33:
            positions.append("top")
        elif center_y > img_h * 0.67:
            positions.append("bottom")
        else:
            positions.append("middle")

        # Horizontal position
        center_x = x + w / 2
        if center_x < img_w * 0.33:
            positions.append("left")
        elif center_x > img_w * 0.67:
            positions.append("right")
        elif "middle" not in positions:
            positions.append("center")

        return positions

    def _describe_texture(self, region: np.ndarray[Any, Any]) -> str | None:
        """Analyze texture/pattern characteristics.

        Args:
            region: Region to analyze

        Returns:
            Texture description or None
        """
        if region.size == 0:
            return None

        # Convert to grayscale for texture analysis
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region

        # Calculate standard deviation as measure of texture
        std_dev = np.std(gray)

        if std_dev < 10:
            return "solid"
        elif std_dev < 30:
            return "smooth"
        elif std_dev < 60:
            return "textured"
        else:
            # Check for edges to distinguish between noisy and structured
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / edges.size

            if edge_ratio > 0.1:
                return "detailed"
            else:
                return "noisy"

    def is_available(self) -> bool:
        """Check if generator is available.

        Returns:
            Always True as this doesn't require external dependencies
        """
        return True

    def get_model_name(self) -> str:
        """Get model name.

        Returns:
            Identifier for this generator
        """
        return "basic-visual-features"
