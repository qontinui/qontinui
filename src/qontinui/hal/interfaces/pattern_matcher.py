"""Pattern matching interface definition."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Match:
    """Represents a pattern match result."""

    x: int
    y: int
    width: int
    height: int
    confidence: float
    center: tuple[int, int]

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Get match bounds as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    @property
    def top_left(self) -> tuple[int, int]:
        """Get top-left corner coordinates."""
        return (self.x, self.y)

    @property
    def bottom_right(self) -> tuple[int, int]:
        """Get bottom-right corner coordinates."""
        return (self.x + self.width, self.y + self.height)


@dataclass
class Feature:
    """Represents a detected feature in an image."""

    x: float
    y: float
    size: float
    angle: float
    response: float
    octave: int
    descriptor: np.ndarray | None = None


class IPatternMatcher(ABC):
    """Interface for pattern matching operations."""

    @abstractmethod
    def find_pattern(
        self,
        haystack: Image.Image,
        needle: Image.Image,
        confidence: float = 0.9,
        grayscale: bool = False,
    ) -> Match | None:
        """Find single pattern occurrence in image.

        Args:
            haystack: Image to search in
            needle: Pattern to search for
            confidence: Minimum confidence threshold (0.0 to 1.0)
            grayscale: Convert to grayscale before matching

        Returns:
            Match object if found, None otherwise
        """
        pass

    @abstractmethod
    def find_all_patterns(
        self,
        haystack: Image.Image,
        needle: Image.Image,
        confidence: float = 0.9,
        grayscale: bool = False,
        limit: int | None = None,
    ) -> list[Match]:
        """Find all pattern occurrences in image.

        Args:
            haystack: Image to search in
            needle: Pattern to search for
            confidence: Minimum confidence threshold
            grayscale: Convert to grayscale before matching
            limit: Maximum number of matches to return

        Returns:
            List of Match objects
        """
        pass

    @abstractmethod
    def find_features(self, image: Image.Image, method: str = "orb") -> list[Feature]:
        """Detect features in image.

        Args:
            image: Image to analyze
            method: Feature detection method ('orb', 'sift', 'surf', etc.)

        Returns:
            List of detected features
        """
        pass

    @abstractmethod
    def match_features(
        self, features1: list[Feature], features2: list[Feature], threshold: float = 0.7
    ) -> list[tuple[Feature, Feature]]:
        """Match features between two feature sets.

        Args:
            features1: First feature set
            features2: Second feature set
            threshold: Matching threshold

        Returns:
            List of matched feature pairs
        """
        pass

    @abstractmethod
    def find_template_multiscale(
        self,
        haystack: Image.Image,
        needle: Image.Image,
        scales: list[float] = None,
        confidence: float = 0.9,
    ) -> Match | None:
        """Find pattern at multiple scales.

        Args:
            haystack: Image to search in
            needle: Pattern to search for
            scales: List of scales to try (default: [0.5, 0.75, 1.0, 1.25, 1.5])
            confidence: Minimum confidence threshold

        Returns:
            Best Match if found, None otherwise
        """
        pass

    @abstractmethod
    def compare_histograms(
        self, image1: Image.Image, image2: Image.Image, method: str = "correlation"
    ) -> float:
        """Compare histograms of two images.

        Args:
            image1: First image
            image2: Second image
            method: Comparison method ('correlation', 'chi-square', 'intersection', 'bhattacharyya')

        Returns:
            Similarity score (0.0 to 1.0)
        """
        pass

    @abstractmethod
    def detect_edges(
        self, image: Image.Image, low_threshold: int = 50, high_threshold: int = 150
    ) -> Image.Image:
        """Detect edges in image.

        Args:
            image: Input image
            low_threshold: Low threshold for edge detection
            high_threshold: High threshold for edge detection

        Returns:
            Edge-detected image
        """
        pass
