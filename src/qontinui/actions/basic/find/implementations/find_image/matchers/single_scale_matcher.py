"""Single-scale template matching implementation."""

import logging
from typing import Any

import cv2
import numpy as np

from .......model.match.match import Match
from ....options.pattern_find_options import PatternFindOptions
from .base_matcher import BaseMatcher

logger = logging.getLogger(__name__)


class SingleScaleMatcher(BaseMatcher):
    """Performs standard template matching at a single scale.

    Uses OpenCV's matchTemplate for efficient correlation-based matching.
    Supports various matching methods (normalized correlation, squared difference, etc.).
    """

    def __init__(self, cv2_method: int) -> None:
        """Initialize single-scale matcher.

        Args:
            cv2_method: OpenCV matching method constant (e.g., cv2.TM_CCOEFF_NORMED)
        """
        self.cv2_method = cv2_method

    def find_matches(
        self,
        template: np.ndarray[Any, Any],
        image: np.ndarray[Any, Any],
        options: PatternFindOptions,
    ) -> list[Match]:
        """Find template matches using single-scale matching.

        Args:
            template: Template image to search for
            image: Image to search within
            options: Pattern matching configuration

        Returns:
            List of matches found
        """
        # Perform template matching
        result = cv2.matchTemplate(image, template, self.cv2_method)

        # Extract template dimensions
        h, w = template.shape[:2]

        matches = []

        if options.search_type.name == "ALL":
            matches = self._find_all_matches(result, w, h, options.similarity)
        else:
            matches = self._find_best_match(result, w, h, options.similarity)

        return matches

    def _find_all_matches(
        self,
        result: np.ndarray[Any, Any],
        width: int,
        height: int,
        threshold: float,
    ) -> list[Match]:
        """Find all matches above threshold.

        Args:
            result: Template matching result matrix
            width: Template width
            height: Template height
            threshold: Minimum similarity threshold

        Returns:
            All matches above threshold
        """
        matches = []
        locations = np.where(result >= threshold)

        for pt in zip(*locations[::-1], strict=False):
            score = float(result[pt[1], pt[0]])
            match = self._create_match(pt[0], pt[1], width, height, score)
            matches.append(match)

        return matches

    def _find_best_match(
        self,
        result: np.ndarray[Any, Any],
        width: int,
        height: int,
        threshold: float,
    ) -> list[Match]:
        """Find single best match.

        Args:
            result: Template matching result matrix
            width: Template width
            height: Template height
            threshold: Minimum similarity threshold

        Returns:
            Best match if above threshold, empty list otherwise
        """
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # For SQDIFF methods, minimum is best match
        if self.cv2_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            confidence = 1 - min_val if self.cv2_method == cv2.TM_SQDIFF_NORMED else 1.0
            top_left = min_loc
        else:
            confidence = max_val
            top_left = max_loc

        if confidence >= threshold:
            match = self._create_match(top_left[0], top_left[1], width, height, confidence)
            return [match]

        return []
