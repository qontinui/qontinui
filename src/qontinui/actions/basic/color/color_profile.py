"""Color profile calculation and matching.

Extracts and compares statistical color information from images.
"""

from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np

from ....model.element.color import HSV


@dataclass
class ColorProfile:
    """Color statistics for an image region.

    Stores statistical information about HSV color distribution
    for use in color matching operations.
    """

    h_min: float
    h_max: float
    h_mean: float
    h_std: float
    s_min: float
    s_max: float
    s_mean: float
    s_std: float
    v_min: float
    v_max: float
    v_mean: float
    v_std: float

    def matches(self, hsv: HSV, std_range: float = 2.0) -> bool:
        """Check if HSV color matches this profile.

        Uses mean and standard deviation to determine if a color
        falls within the expected range for this profile.

        Args:
            hsv: Color to check
            std_range: Number of standard deviations for tolerance

        Returns:
            True if color matches profile
        """
        # Check hue (circular)
        h_low = (self.h_mean - self.h_std * std_range) % 360
        h_high = (self.h_mean + self.h_std * std_range) % 360
        if h_low > h_high:  # Wraps around 0
            h_matches = hsv.hue >= h_low or hsv.hue <= h_high
        else:
            h_matches = h_low <= hsv.hue <= h_high

        # Check saturation
        s_low = max(0, self.s_mean - self.s_std * std_range)
        s_high = min(255, self.s_mean + self.s_std * std_range)
        s_matches = s_low <= hsv.saturation <= s_high

        # Check value
        v_low = max(0, self.v_mean - self.v_std * std_range)
        v_high = min(255, self.v_mean + self.v_std * std_range)
        v_matches = v_low <= hsv.value <= v_high

        return cast(bool, h_matches and s_matches and v_matches)


class ColorProfileCalculator:
    """Calculates color profiles from images.

    Provides methods to extract statistical color information
    with optional caching for performance.
    """

    def __init__(self) -> None:
        """Initialize calculator with empty cache."""
        self._cache: dict[str, ColorProfile] = {}

    def calculate(self, image: np.ndarray[Any, Any]) -> ColorProfile:
        """Calculate color statistics for image.

        Args:
            image: Input image in BGR format

        Returns:
            Color profile with statistics
        """
        # Check cache
        cache_key = str(image.shape) + str(image.sum())
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Calculate statistics for each channel
        h_channel = hsv[:, :, 0].flatten()
        s_channel = hsv[:, :, 1].flatten()
        v_channel = hsv[:, :, 2].flatten()

        profile = ColorProfile(
            h_min=float(np.min(h_channel)),
            h_max=float(np.max(h_channel)),
            h_mean=float(np.mean(h_channel)),
            h_std=float(np.std(h_channel)),
            s_min=float(np.min(s_channel)),
            s_max=float(np.max(s_channel)),
            s_mean=float(np.mean(s_channel)),
            s_std=float(np.std(s_channel)),
            v_min=float(np.min(v_channel)),
            v_max=float(np.max(v_channel)),
            v_mean=float(np.mean(v_channel)),
            v_std=float(np.std(v_channel)),
        )

        # Cache result
        self._cache[cache_key] = profile

        return profile

    def clear_cache(self) -> None:
        """Clear the profile cache."""
        self._cache.clear()

    def calculate_score(self, hsv: HSV, profile: ColorProfile) -> float:
        """Calculate how well a color matches a profile.

        Args:
            hsv: Color to score
            profile: Target profile

        Returns:
            Score (0.0-1.0)
        """
        # Calculate normalized distances
        h_dist = abs(hsv.hue - profile.h_mean) / 180.0
        s_dist = abs(hsv.saturation - profile.s_mean) / 255.0
        v_dist = abs(hsv.value - profile.v_mean) / 255.0

        # Combined score (inverse of distance)
        avg_dist = (h_dist + s_dist + v_dist) / 3.0
        return cast(float, max(0.0, 1.0 - avg_dist))
