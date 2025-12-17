"""Color statistics utilities - ported from Qontinui framework.

Statistical analysis of color distributions for MU strategy.
"""

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ....model.element.color import HSV

logger = logging.getLogger(__name__)


@dataclass
class ColorStatistics:
    """Statistical color profile for an image region.

    Port of MU from Qontinui framework (mean/mu) color statistics.

    Captures min, max, mean, and standard deviation of color channels.
    """

    # HSV statistics
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

    # RGB statistics (optional)
    r_mean: float | None = None
    g_mean: float | None = None
    b_mean: float | None = None

    def matches_hsv(self, hsv: HSV, tolerance_std: float = 2.0) -> bool:
        """Check if an HSV color matches this profile.

        Args:
            hsv: Color to check
            tolerance_std: Number of standard deviations for tolerance

        Returns:
            True if color matches profile
        """
        # Check hue (handle circular nature)
        h_low = (self.h_mean - self.h_std * tolerance_std) % 360
        h_high = (self.h_mean + self.h_std * tolerance_std) % 360

        if h_low > h_high:  # Wraps around 0
            h_matches = hsv.hue >= h_low or hsv.hue <= h_high
        else:
            h_matches = h_low <= hsv.hue <= h_high

        # Check saturation
        s_low = max(0, self.s_mean - self.s_std * tolerance_std)
        s_high = min(255, self.s_mean + self.s_std * tolerance_std)
        s_matches = s_low <= hsv.saturation <= s_high

        # Check value
        v_low = max(0, self.v_mean - self.v_std * tolerance_std)
        v_high = min(255, self.v_mean + self.v_std * tolerance_std)
        v_matches = v_low <= hsv.value <= v_high

        return h_matches and s_matches and v_matches

    def similarity_score(self, other: "ColorStatistics") -> float:
        """Calculate similarity to another color profile.

        Args:
            other: Profile to compare

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Calculate normalized differences for each channel
        h_diff = min(abs(self.h_mean - other.h_mean), 360 - abs(self.h_mean - other.h_mean)) / 180.0
        s_diff = abs(self.s_mean - other.s_mean) / 255.0
        v_diff = abs(self.v_mean - other.v_mean) / 255.0

        # Weight hue less as it's more variable
        avg_diff = h_diff * 0.3 + s_diff * 0.35 + v_diff * 0.35

        return max(0.0, 1.0 - avg_diff)


@dataclass
class ColorStatisticsAnalyzer:
    """Analyzes color statistics for image regions.

    Port of MU from Qontinui framework strategy analyzer.

    Creates statistical profiles of color distributions.
    """

    include_rgb: bool = False  # Whether to include RGB statistics

    def analyze_image(self, image: np.ndarray[Any, Any]) -> ColorStatistics:
        """Calculate color statistics for an image.

        Args:
            image: Image to analyze (BGR format)

        Returns:
            ColorStatistics profile
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Split channels
        h_channel = hsv[:, :, 0].flatten()
        s_channel = hsv[:, :, 1].flatten()
        v_channel = hsv[:, :, 2].flatten()

        # Calculate HSV statistics
        stats = ColorStatistics(
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

        # Optionally calculate RGB statistics
        if self.include_rgb:
            b_channel = image[:, :, 0].flatten()
            g_channel = image[:, :, 1].flatten()
            r_channel = image[:, :, 2].flatten()

            stats.r_mean = float(np.mean(r_channel))
            stats.g_mean = float(np.mean(g_channel))
            stats.b_mean = float(np.mean(b_channel))

        logger.debug(
            f"Color statistics: H={stats.h_mean:.1f}±{stats.h_std:.1f}, "
            f"S={stats.s_mean:.1f}±{stats.s_std:.1f}, "
            f"V={stats.v_mean:.1f}±{stats.v_std:.1f}"
        )

        return stats

    def create_color_mask(
        self,
        image: np.ndarray[Any, Any],
        stats: ColorStatistics,
        tolerance_std: float = 2.0,
    ) -> np.ndarray[Any, Any]:
        """Create a binary mask for pixels matching color statistics.

        Args:
            image: Image to process
            stats: Color statistics to match
            tolerance_std: Standard deviation tolerance

        Returns:
            Binary mask (255 for matching pixels, 0 for non-matching)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Calculate tolerance ranges
        h_low = (stats.h_mean - stats.h_std * tolerance_std) % 180
        h_high = (stats.h_mean + stats.h_std * tolerance_std) % 180

        s_low = max(0, stats.s_mean - stats.s_std * tolerance_std)
        s_high = min(255, stats.s_mean + stats.s_std * tolerance_std)

        v_low = max(0, stats.v_mean - stats.v_std * tolerance_std)
        v_high = min(255, stats.v_mean + stats.v_std * tolerance_std)

        # Create mask using inRange
        if h_low > h_high:  # Hue wraps around
            mask1 = cv2.inRange(
                hsv, np.array([0, s_low, v_low]), np.array([h_high, s_high, v_high])
            )
            mask2 = cv2.inRange(
                hsv, np.array([h_low, s_low, v_low]), np.array([180, s_high, v_high])
            )
            mask = cv2.bitwise_or(mask1, mask2).astype(np.uint8)
        else:
            mask = cv2.inRange(
                hsv, np.array([h_low, s_low, v_low]), np.array([h_high, s_high, v_high])
            ).astype(np.uint8)

        return mask
