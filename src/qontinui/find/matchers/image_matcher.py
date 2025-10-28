"""Abstract base interface for image matching algorithms.

Defines the contract that all image matchers must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

from ...model.element import Pattern
from ..match import Match


class ImageMatcher(ABC):
    """Abstract base class for image matching algorithms.

    All concrete matcher implementations must inherit from this class
    and implement the find_matches method. This provides a common interface
    for different matching strategies (template matching, feature matching, AI-based, etc.).
    """

    @abstractmethod
    def find_matches(
        self,
        screenshot: Any,
        pattern: Pattern,
        find_all: bool = False,
        similarity: float = 0.8,
        search_region: tuple[int, int, int, int] | None = None,
    ) -> list[Match]:
        """Find pattern matches in the screenshot.

        Args:
            screenshot: Screenshot image (PIL Image, numpy array, or OpenCV mat)
            pattern: Pattern to search for with pixel data and optional mask
            find_all: If True, find all matches above threshold. If False, return only best match
            similarity: Minimum similarity threshold (0.0 to 1.0)
            search_region: Optional region to search in as (x, y, width, height).
                          If provided, coordinates are relative to this region.

        Returns:
            List of Match objects found. Empty list if no matches meet threshold.
            Matches are sorted by similarity score (highest first).

        Raises:
            ImageProcessingError: If image conversion or matching fails
        """
        pass
