"""OCR engine interface definition."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass
class TextRegion:
    """Represents a text region detected in an image."""

    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    language: str | None = None

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Get region bounds as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    @property
    def center(self) -> tuple[int, int]:
        """Get center point of text region."""
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class TextMatch:
    """Represents a text match result."""

    text: str
    region: TextRegion
    similarity: float


class IOCREngine(ABC):
    """Interface for OCR operations."""

    @abstractmethod
    def extract_text(
        self, image: Image.Image, languages: list[str] | None = None
    ) -> str:
        """Extract all text from image.

        Args:
            image: Image to extract text from
            languages: List of language codes (e.g., ['en', 'es'])

        Returns:
            Extracted text as string
        """
        pass

    @abstractmethod
    def get_text_regions(
        self,
        image: Image.Image,
        languages: list[str] | None = None,
        min_confidence: float = 0.5,
    ) -> list[TextRegion]:
        """Get all text regions with bounding boxes.

        Args:
            image: Image to analyze
            languages: List of language codes
            min_confidence: Minimum confidence threshold

        Returns:
            List of TextRegion objects
        """
        pass

    @abstractmethod
    def find_text(
        self,
        image: Image.Image,
        text: str,
        case_sensitive: bool = False,
        confidence: float = 0.8,
    ) -> TextMatch | None:
        """Find specific text in image.

        Args:
            image: Image to search
            text: Text to find
            case_sensitive: Whether search is case-sensitive
            confidence: Minimum confidence threshold

        Returns:
            TextMatch if found, None otherwise
        """
        pass

    @abstractmethod
    def find_all_text(
        self,
        image: Image.Image,
        text: str,
        case_sensitive: bool = False,
        confidence: float = 0.8,
    ) -> list[TextMatch]:
        """Find all occurrences of text in image.

        Args:
            image: Image to search
            text: Text to find
            case_sensitive: Whether search is case-sensitive
            confidence: Minimum confidence threshold

        Returns:
            List of TextMatch objects
        """
        pass

    @abstractmethod
    def extract_text_from_region(
        self,
        image: Image.Image,
        region: tuple[int, int, int, int],
        languages: list[str] | None = None,
    ) -> str:
        """Extract text from specific region.

        Args:
            image: Image containing text
            region: Region bounds (x, y, width, height)
            languages: List of language codes

        Returns:
            Extracted text from region
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes.

        Returns:
            List of language codes (e.g., ['en', 'es', 'fr'])
        """
        pass

    @abstractmethod
    def preprocess_image(
        self,
        image: Image.Image,
        grayscale: bool = True,
        denoise: bool = True,
        threshold: bool = False,
    ) -> Image.Image:
        """Preprocess image for better OCR results.

        Args:
            image: Input image
            grayscale: Convert to grayscale
            denoise: Apply denoising
            threshold: Apply thresholding

        Returns:
            Preprocessed image
        """
        pass

    @abstractmethod
    def detect_text_orientation(self, image: Image.Image) -> dict[str, Any]:
        """Detect text orientation in image.

        Args:
            image: Image to analyze

        Returns:
            Dictionary with orientation info:
                - angle: Rotation angle in degrees
                - confidence: Confidence score
                - script: Detected script type
        """
        pass
