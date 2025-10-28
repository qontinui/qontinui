"""Region extraction from binary masks.

Extracts contiguous regions using contour detection and converts
them into Match objects.
"""

from typing import Any

import cv2
import numpy as np

from ....model.element.location import Location
from ....model.element.region import Region
from ....model.match.match import Match


class RegionExtractor:
    """Extracts regions from binary masks using contour detection.

    Provides utilities for finding contiguous regions in images
    and converting them to Match objects.
    """

    @staticmethod
    def extract_from_mask(
        mask: np.ndarray[Any, Any], min_size: int = 0, default_score: float = 0.8, name: str = ""
    ) -> list[Match]:
        """Extract regions from a binary mask.

        Finds contours in the mask and creates Match objects for each
        region that meets the size requirement.

        Args:
            mask: Binary mask (0 or 255)
            min_size: Minimum width and height for valid regions
            default_score: Score to assign to matches
            name: Name for the matches

        Returns:
            List of matches representing found regions
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        matches = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_size and h >= min_size:
                region = Region(x, y, w, h)
                match = Match(score=default_score, target=Location(region=region), name=name)
                matches.append(match)

        return matches

    @staticmethod
    def extract_from_classification(
        classification: np.ndarray[Any, Any], class_id: int, default_score: float = 0.9
    ) -> list[Match]:
        """Extract regions for a specific class from classification result.

        Creates a binary mask for the given class and extracts regions.

        Args:
            classification: Classification result array with class IDs
            class_id: Class ID to extract regions for
            default_score: Score to assign to matches

        Returns:
            List of matches for the specified class
        """
        # Create binary mask for this class
        mask = (classification == class_id).astype(np.uint8) * 255

        # Extract regions
        return RegionExtractor.extract_from_mask(
            mask, min_size=0, default_score=default_score, name=f"class_{class_id}"
        )

    @staticmethod
    def filter_by_area(
        matches: list[Match], min_area: int = 1, max_area: int = -1
    ) -> list[Match]:
        """Filter matches by area constraints.

        Args:
            matches: Matches to filter
            min_area: Minimum area in pixels
            max_area: Maximum area in pixels (-1 for no limit)

        Returns:
            Filtered matches
        """
        filtered = []

        for match in matches:
            if not match.region:
                continue

            area = match.region.width * match.region.height

            # Check minimum area
            if area < min_area:
                continue

            # Check maximum area
            if max_area > 0 and area > max_area:
                continue

            filtered.append(match)

        return filtered
