"""Match image operations.

Provides image manipulation and extraction utilities for Match objects.
"""


from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .match import Match


class MatchImageOps:
    """Provides image operations for Match objects."""

    @staticmethod
    def get_mat(match: Match) -> np.ndarray[Any, Any] | None:
        """Get the image as BGR NumPy array.

        Args:
            match: Match object

        Returns:
            BGR array or None
        """
        return match.image.get_mat_bgr() if match.image else None

    @staticmethod
    def set_image_from_scene(match: Match) -> None:
        """Set image from scene if available.

        Args:
            match: Match object to update
        """
        if match.metadata.scene is None:
            return

        # Extract sub-image from scene
        # This would need implementation of BufferedImageUtilities
        # For now, just a placeholder
        pass
