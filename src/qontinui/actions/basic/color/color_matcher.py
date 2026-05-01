"""Color matching utilities.

Provides functions for matching colors within tolerance ranges
and creating color masks.
"""

from typing import Any

import cv2
import numpy as np

from ....model.element.color import HSV, RGB
from ....model.element.location import Location
from ....model.element.region import Region
from ....model.match.match import Match


class ColorMatcher:
    """Utilities for color-based matching operations.

    Provides methods for creating color masks and finding regions
    that match specific colors within tolerance ranges.
    """

    @staticmethod
    def create_hsv_mask(
        scene: np.ndarray[Any, Any], color: RGB, tolerance: int = 30
    ) -> np.ndarray[Any, Any]:
        """Create a binary mask for pixels matching a color.

        Args:
            scene: Scene image in BGR format
            color: Target color in RGB
            tolerance: Color tolerance in HSV space

        Returns:
            Binary mask (0 or 255)
        """
        # Convert color to HSV for better matching
        target_hsv = color.to_hsv()

        # Create color range
        lower = np.array(
            [
                max(0, target_hsv.hue - tolerance),
                max(0, target_hsv.saturation - tolerance),
                max(0, target_hsv.value - tolerance),
            ]
        )
        upper = np.array(
            [
                min(179, target_hsv.hue + tolerance),
                min(255, target_hsv.saturation + tolerance),
                min(255, target_hsv.value + tolerance),
            ]
        )

        # Convert scene to HSV
        scene_hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)

        # Create mask
        return cv2.inRange(scene_hsv, lower, upper)

    @staticmethod
    def find_color_regions(
        scene: np.ndarray[Any, Any], color: RGB, min_size: int, tolerance: int = 30
    ) -> list[Match]:
        """Find regions matching a specific color.

        Args:
            scene: Scene to search in BGR format
            color: Target color in RGB
            min_size: Minimum region size (width and height)
            tolerance: Color tolerance in HSV space

        Returns:
            List of matches
        """
        # Create mask
        mask = ColorMatcher.create_hsv_mask(scene, color, tolerance)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        matches = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_size and h >= min_size:
                region = Region(x, y, w, h)
                match = Match(score=0.85, target=Location(region=region))
                matches.append(match)

        return matches

    @staticmethod
    def create_profile_mask(
        scene: np.ndarray[Any, Any], profile: Any, std_range: float = 2.0
    ) -> np.ndarray[Any, Any]:
        """Create a binary mask for pixels matching a color profile.

        Args:
            scene: Scene image in BGR format
            profile: ColorProfile to match against
            std_range: Number of standard deviations for tolerance

        Returns:
            Binary mask (0 or 255)
        """
        h, w = scene.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        scene_hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)

        # Check each pixel against profile
        for y in range(h):
            for x in range(w):
                pixel_hsv = HSV(
                    scene_hsv[y, x, 0], scene_hsv[y, x, 1], scene_hsv[y, x, 2]
                )
                if profile.matches(pixel_hsv, std_range):
                    mask[y, x] = 255

        return mask
