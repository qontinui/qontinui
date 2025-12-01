"""Multi-class classification color strategy.

Performs pixel-by-pixel classification using color profiles from
multiple classes (targets and context).
"""

import logging
from typing import Any

import cv2
import numpy as np

from .....model.element.color import HSV
from .....model.match.match import Match
from ..color_profile import ColorProfileCalculator
from ..region_extractor import RegionExtractor
from .base_strategy import BaseColorStrategy

logger = logging.getLogger(__name__)


class ClassificationStrategy(BaseColorStrategy):
    """Multi-class classification strategy for color finding.

    Builds color profiles for multiple classes, then classifies each
    pixel in the scene and extracts contiguous regions.
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
        """Find color regions using multi-class classification.

        Args:
            scene: Scene image to search
            target_images: Target and context images for classification
            options: Color finding options

        Returns:
            List of matches
        """
        if not target_images:
            logger.warning("No images for classification")
            return []

        # Build color profiles for each class
        class_profiles = {}
        for i, img in enumerate(target_images):
            class_profiles[i] = self._profile_calculator.calculate(img)

        # Classify each pixel in scene
        classification = self._classify_scene(scene, class_profiles)

        # Extract contiguous regions for each class
        matches = []
        for class_id in class_profiles.keys():
            if class_id == -1:
                continue

            class_matches = RegionExtractor.extract_from_classification(
                classification, class_id, default_score=0.9
            )
            matches.extend(class_matches)

        return matches

    def _classify_scene(
        self, scene: np.ndarray[Any, Any], class_profiles: dict[int, Any]
    ) -> np.ndarray[Any, Any]:
        """Classify each pixel in the scene.

        Args:
            scene: Scene image to classify
            class_profiles: Dictionary of class profiles

        Returns:
            Classification array with class IDs
        """
        h, w = scene.shape[:2]
        classification = np.zeros((h, w), dtype=np.int32)

        scene_hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)

        for y in range(h):
            for x in range(w):
                pixel_hsv = HSV(
                    scene_hsv[y, x, 0], scene_hsv[y, x, 1], scene_hsv[y, x, 2]
                )

                # Find best matching class
                best_class = -1
                best_score = 0.0

                for class_id, profile in class_profiles.items():
                    if profile.matches(pixel_hsv):
                        # Calculate score based on distance from mean
                        score = self._profile_calculator.calculate_score(
                            pixel_hsv, profile
                        )
                        if score > best_score:
                            best_score = score
                            best_class = class_id

                classification[y, x] = best_class

        return classification
