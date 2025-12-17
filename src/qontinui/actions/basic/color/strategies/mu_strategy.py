"""Mean/standard deviation color strategy.

Uses statistical color profiles (mean and standard deviation) to match
regions in the scene.
"""

import logging
from typing import Any

import numpy as np

from .....model.match.match import Match
from ..color_matcher import ColorMatcher
from ..color_profile import ColorProfileCalculator
from ..region_extractor import RegionExtractor
from .base_strategy import BaseColorStrategy

logger = logging.getLogger(__name__)


class MUStrategy(BaseColorStrategy):
    """Mean/standard deviation strategy for color finding.

    Calculates statistical color profiles from target images,
    then finds regions in the scene matching those profiles.
    """

    def __init__(self) -> None:
        """Initialize strategy with profile calculator."""
        self._profile_calculator = ColorProfileCalculator()

    def find_color_regions(
        self,
        scene: np.ndarray[Any, Any],
        target_images: list[np.ndarray[Any, Any]],
        options: Any,
    ) -> list[Match]:
        """Find color regions using mean/std color statistics.

        Args:
            scene: Scene image to search
            target_images: Target images containing desired colors
            options: Color finding options

        Returns:
            List of matches
        """
        matches = []

        if not target_images:
            logger.warning("No target images for MU color finding")
            return []

        for target in target_images:
            # Get color profile from target
            profile = self._profile_calculator.calculate(target)

            # Find regions matching profile
            profile_matches = self._find_profile_regions(
                scene, profile, options.get_diameter()
            )
            matches.extend(profile_matches)

        return matches

    def _find_profile_regions(
        self, scene: np.ndarray[Any, Any], profile: Any, min_size: int
    ) -> list[Match]:
        """Find regions matching a color profile.

        Args:
            scene: Scene to search
            profile: Target color profile
            min_size: Minimum region size

        Returns:
            List of matches
        """
        # Create mask using profile
        mask = ColorMatcher.create_profile_mask(scene, profile)

        # Extract regions from mask
        return RegionExtractor.extract_from_mask(
            mask, min_size=min_size, default_score=0.8
        )
