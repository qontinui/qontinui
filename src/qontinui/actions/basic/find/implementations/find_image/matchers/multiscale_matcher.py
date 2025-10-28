"""Multi-scale template matching implementation."""

import logging
from typing import Any

import cv2
import numpy as np

from .......model.match.match import Match
from ....options.pattern_find_options import PatternFindOptions
from .base_matcher import BaseMatcher
from .single_scale_matcher import SingleScaleMatcher

logger = logging.getLogger(__name__)


class MultiScaleMatcher(BaseMatcher):
    """Performs scale-invariant template matching.

    Searches for the template at multiple scales to handle size variations.
    Useful when the target may appear at different sizes than the template.
    """

    def __init__(self, cv2_method: int) -> None:
        """Initialize multi-scale matcher.

        Args:
            cv2_method: OpenCV matching method constant
        """
        self.cv2_method = cv2_method
        self.single_scale_matcher = SingleScaleMatcher(cv2_method)

    def find_matches(
        self,
        template: np.ndarray[Any, Any],
        image: np.ndarray[Any, Any],
        options: PatternFindOptions,
    ) -> list[Match]:
        """Find template matches at multiple scales.

        Args:
            template: Template image to search for
            image: Image to search within
            options: Pattern matching configuration with scale parameters

        Returns:
            List of matches found across all scales
        """
        all_matches = []

        # Iterate through scale range
        scale = options.min_scale
        while scale <= options.max_scale:
            # Resize template to current scale
            scaled_template = self._resize_template(template, scale)

            if scaled_template is None:
                scale += options.scale_step
                continue

            # Find matches at this scale
            matches = self.single_scale_matcher.find_matches(scaled_template, image, options)

            all_matches.extend(matches)

            # Early termination if excellent match found
            if self._should_terminate_early(matches, options.early_termination_threshold):
                logger.debug(f"Early termination at scale {scale:.2f}")
                break

            scale += options.scale_step

        return all_matches

    def _resize_template(
        self,
        template: np.ndarray[Any, Any],
        scale: float,
    ) -> np.ndarray[Any, Any] | None:
        """Resize template to specified scale.

        Args:
            template: Original template image
            scale: Scale factor to apply

        Returns:
            Resized template or None if dimensions too small
        """
        new_width = int(template.shape[1] * scale)
        new_height = int(template.shape[0] * scale)

        # Skip if template would be too small
        if new_width < 10 or new_height < 10:
            logger.debug(f"Skipping scale {scale:.2f} - template too small")
            return None

        return cv2.resize(template, (new_width, new_height))

    def _should_terminate_early(
        self,
        matches: list[Match],
        threshold: float,
    ) -> bool:
        """Check if early termination conditions are met.

        Args:
            matches: Matches found at current scale
            threshold: Early termination threshold (0 = disabled)

        Returns:
            True if should terminate, False otherwise
        """
        if threshold <= 0:
            return False

        return any(m.similarity >= threshold for m in matches)
