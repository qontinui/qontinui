"""Position type classification for StateImages.

This module provides functionality for determining whether StateImages appear at
fixed positions (same location across frames) or dynamic positions (varying
locations). This is important for distinguishing between static UI elements
and dynamic content.

Key Features:
- Template matching across multiple frames
- Position variance analysis
- Fixed vs. dynamic classification
- Occurrence tracking

Example:
    >>> from qontinui.discovery.pixel_analysis import PositionClassifier
    >>> classifier = PositionClassifier()
    >>> is_fixed = classifier.is_position_fixed(state_image, frames)
    >>> print(f"Position type: {'fixed' if is_fixed else 'dynamic'}")
"""

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PositionClassifier:
    """Classifies StateImage position types as fixed or dynamic.

    This classifier analyzes multiple occurrences of a StateImage across frames
    to determine if it appears at a consistent location (fixed) or varies
    across different positions (dynamic).

    Fixed positions indicate static UI elements like:
    - Menu bars
    - Toolbars
    - Status bars
    - Fixed buttons

    Dynamic positions indicate content that moves:
    - Draggable windows
    - List items
    - Dynamic content
    """

    def __init__(
        self,
        position_tolerance_px: int = 5,
        similarity_threshold: float = 0.85,
    ) -> None:
        """Initialize the position classifier.

        Args:
            position_tolerance_px: Maximum pixel distance for positions to be
                considered "fixed". Positions within this tolerance are
                considered the same location.
            similarity_threshold: Minimum similarity score for template matching.
        """
        self.position_tolerance_px = position_tolerance_px
        self.similarity_threshold = similarity_threshold

    def is_position_fixed(
        self,
        image: np.ndarray[Any, Any],
        frames: list[np.ndarray[Any, Any]],
    ) -> bool:
        """Determine if an image appears at a fixed position across frames.

        Analyzes multiple occurrences to determine if the image appears at
        the same location across frames (fixed) or at varying locations (dynamic).

        Args:
            image: The image to search for (template).
            frames: List of frames to search in.

        Returns:
            True if position is fixed (same location across frames), False otherwise.
        """
        occurrences = self.find_occurrences(image, frames)

        if len(occurrences) < 2:
            # Not enough data, assume fixed if found only once
            return True

        # Calculate variance in positions
        positions = np.array(occurrences)
        mean_pos = positions.mean(axis=0)
        distances = np.sqrt(((positions - mean_pos) ** 2).sum(axis=1))
        max_distance = distances.max()

        # If all occurrences are within tolerance, consider it fixed
        is_fixed = max_distance <= self.position_tolerance_px

        logger.debug(
            f"Position analysis: {len(occurrences)} occurrences, "
            f"max distance: {max_distance:.2f}, fixed: {is_fixed}"
        )

        return bool(is_fixed)

    def find_occurrences(
        self,
        template: np.ndarray[Any, Any],
        frames: list[np.ndarray[Any, Any]],
    ) -> list[tuple[int, int]]:
        """Find all occurrences of a template across multiple frames.

        Uses template matching to locate the template in each frame. Returns
        the top-left coordinates of each match.

        Args:
            template: The template image to search for.
            frames: List of frames to search in.

        Returns:
            List of (x, y) positions where the template was found.
        """
        occurrences = []

        # Convert template to grayscale for matching
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template

        for frame_idx, frame in enumerate(frames):
            try:
                # Convert frame to grayscale
                if len(frame.shape) == 3:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame_gray = frame

                # Perform template matching
                result = cv2.matchTemplate(
                    frame_gray,
                    template_gray,
                    cv2.TM_CCOEFF_NORMED,
                )

                # Find locations above threshold
                locations = np.where(result >= self.similarity_threshold)

                # Add locations (convert from (y, x) to (x, y))
                for y, x in zip(*locations, strict=False):
                    occurrences.append((x, y))

            except Exception as e:
                logger.debug(f"Error matching in frame {frame_idx}: {e}")

        return occurrences

    def classify_position_type(
        self,
        image: np.ndarray[Any, Any],
        frames: list[np.ndarray[Any, Any]],
    ) -> str:
        """Classify position type as "fixed" or "dynamic".

        Convenience method that returns a string classification.

        Args:
            image: The image to analyze.
            frames: List of frames to analyze.

        Returns:
            "fixed" if position is fixed, "dynamic" otherwise.
        """
        is_fixed = self.is_position_fixed(image, frames)
        return "fixed" if is_fixed else "dynamic"

    def get_position_statistics(
        self,
        image: np.ndarray[Any, Any],
        frames: list[np.ndarray[Any, Any]],
    ) -> dict[str, Any]:
        """Get detailed position statistics.

        Returns comprehensive statistics about position variance including
        occurrence count, mean position, max distance, and classification.

        Args:
            image: The image to analyze.
            frames: List of frames to analyze.

        Returns:
            Dictionary with position statistics:
                - occurrences_count: Number of times image was found
                - mean_position: Average (x, y) position
                - max_distance: Maximum distance from mean position
                - position_type: "fixed" or "dynamic"
                - is_fixed: Boolean classification
        """
        occurrences = self.find_occurrences(image, frames)

        if not occurrences:
            return {
                "occurrences_count": 0,
                "mean_position": None,
                "max_distance": 0.0,
                "position_type": "unknown",
                "is_fixed": False,
            }

        positions = np.array(occurrences)
        mean_pos = positions.mean(axis=0)
        distances = np.sqrt(((positions - mean_pos) ** 2).sum(axis=1))
        max_distance = float(distances.max())

        is_fixed = max_distance <= self.position_tolerance_px

        return {
            "occurrences_count": len(occurrences),
            "mean_position": tuple(mean_pos.astype(int)),
            "max_distance": max_distance,
            "position_type": "fixed" if is_fixed else "dynamic",
            "is_fixed": is_fixed,
        }
