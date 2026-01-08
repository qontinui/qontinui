"""Base analyzer class for environment discovery.

Provides common functionality for all environment analyzers including
logging, progress tracking, and confidence calculation.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseAnalyzer(ABC, Generic[T]):
    """Abstract base class for environment analyzers.

    Each analyzer extracts specific visual characteristics from screenshots.
    Subclasses must implement the analyze() method.

    Attributes:
        name: Human-readable name for this analyzer.
        confidence: Confidence score for the last analysis (0.0-1.0).
    """

    def __init__(self, name: str) -> None:
        """Initialize the analyzer.

        Args:
            name: Human-readable name for this analyzer.
        """
        self.name = name
        self.confidence: float = 0.0
        self._screenshots_analyzed: int = 0

    @abstractmethod
    async def analyze(
        self,
        screenshots: list[NDArray[np.uint8]],
        **kwargs: Any,
    ) -> T:
        """Analyze screenshots and extract characteristics.

        Args:
            screenshots: List of screenshots as numpy arrays (BGR format).
            **kwargs: Additional analyzer-specific parameters.

        Returns:
            Extracted characteristics in analyzer-specific format.
        """
        pass

    def reset(self) -> None:
        """Reset analyzer state for fresh analysis."""
        self.confidence = 0.0
        self._screenshots_analyzed = 0

    def _calculate_confidence(
        self,
        sample_count: int,
        min_samples: int = 3,
        optimal_samples: int = 10,
        consistency_score: float = 1.0,
    ) -> float:
        """Calculate confidence score based on sample count and consistency.

        Args:
            sample_count: Number of samples analyzed.
            min_samples: Minimum samples for non-zero confidence.
            optimal_samples: Sample count for maximum confidence.
            consistency_score: How consistent results are (0.0-1.0).

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if sample_count < min_samples:
            sample_factor = 0.0
        elif sample_count >= optimal_samples:
            sample_factor = 1.0
        else:
            sample_factor = (sample_count - min_samples) / (optimal_samples - min_samples)

        return min(1.0, sample_factor * consistency_score)

    def _log_progress(self, message: str, level: int = logging.INFO) -> None:
        """Log progress message with analyzer context.

        Args:
            message: Message to log.
            level: Logging level (default INFO).
        """
        logger.log(level, f"[{self.name}] {message}")

    @staticmethod
    def _ensure_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Ensure image is in BGR format (3 channels).

        Args:
            image: Input image array.

        Returns:
            Image in BGR format.
        """
        if len(image.shape) == 2:
            # Grayscale to BGR
            return np.stack([image, image, image], axis=-1)
        elif image.shape[2] == 4:
            # BGRA to BGR
            return image[:, :, :3]
        return image

    @staticmethod
    def _bgr_to_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Convert BGR image to RGB.

        Args:
            image: BGR image array.

        Returns:
            RGB image array.
        """
        return image[:, :, ::-1]

    @staticmethod
    def _rgb_to_hex(r: int, g: int, b: int) -> str:
        """Convert RGB values to hex color string.

        Args:
            r: Red value (0-255).
            g: Green value (0-255).
            b: Blue value (0-255).

        Returns:
            Hex color string (e.g., "#FF5733").
        """
        return f"#{r:02X}{g:02X}{b:02X}"

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        """Convert hex color string to RGB tuple.

        Args:
            hex_color: Hex color string (e.g., "#FF5733").

        Returns:
            Tuple of (r, g, b) values.
        """
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore

    @staticmethod
    def _calculate_luminance(r: int, g: int, b: int) -> float:
        """Calculate relative luminance of a color.

        Uses the formula from WCAG 2.0.

        Args:
            r: Red value (0-255).
            g: Green value (0-255).
            b: Blue value (0-255).

        Returns:
            Relative luminance (0.0-1.0).
        """

        def adjust(c: int) -> float:
            c_norm = c / 255.0
            if c_norm <= 0.03928:
                return c_norm / 12.92
            return ((c_norm + 0.055) / 1.055) ** 2.4

        return 0.2126 * adjust(r) + 0.7152 * adjust(g) + 0.0722 * adjust(b)
