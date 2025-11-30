"""Vision and perception-related exceptions.

This module contains exceptions for element finding, image recognition,
text detection, and other vision-based operations.
"""

from .base_exceptions import QontinuiException


class PerceptionException(QontinuiException):
    """Base exception for perception/matching errors."""

    pass


class ElementNotFoundException(PerceptionException):
    """Raised when element cannot be found."""

    def __init__(
        self, element_description: str, search_region: str | None = None, **kwargs
    ) -> None:
        """Initialize with search details."""
        message = f"Element '{element_description}' not found"
        if search_region:
            message += f" in region {search_region}"

        super().__init__(
            message,
            error_code="ELEMENT_NOT_FOUND",
            context={"element": element_description, "search_region": search_region, **kwargs},
        )


class ImageNotFoundException(ElementNotFoundException):
    """Raised when image cannot be found on screen."""

    def __init__(self, image_path: str, similarity: float, **kwargs) -> None:
        """Initialize with image search details."""
        super().__init__(f"Image '{image_path}' (similarity: {similarity})", **kwargs)
        self.context["similarity"] = similarity


class TextNotFoundException(ElementNotFoundException):
    """Raised when text cannot be found on screen."""

    def __init__(self, text: str, **kwargs) -> None:
        """Initialize with text search details."""
        super().__init__(f"Text '{text}'", **kwargs)


class AmbiguousMatchException(PerceptionException):
    """Raised when multiple matches found but only one expected."""

    def __init__(self, element: str, match_count: int, **kwargs) -> None:
        """Initialize with match details."""
        super().__init__(
            f"Found {match_count} matches for '{element}', expected 1",
            error_code="AMBIGUOUS_MATCH",
            context={"element": element, "match_count": match_count, **kwargs},
        )


class InvalidImageException(PerceptionException):
    """Raised when image is invalid or corrupted."""

    def __init__(self, image_path: str, reason: str, **kwargs) -> None:
        """Initialize with image details."""
        super().__init__(
            f"Invalid image '{image_path}': {reason}",
            error_code="INVALID_IMAGE",
            context={"image_path": image_path, "reason": reason, **kwargs},
        )


class PatternMatchError(PerceptionException):
    """Raised when pattern matching operation fails."""

    def __init__(self, reason: str, pattern_path: str | None = None, **kwargs) -> None:
        """Initialize with pattern matching details."""
        message = "Pattern matching failed"
        if pattern_path:
            message += f" for pattern '{pattern_path}'"
        message += f": {reason}"

        super().__init__(
            message,
            error_code="PATTERN_MATCH_FAILED",
            context={"reason": reason, "pattern_path": pattern_path, **kwargs},
        )


class OCRError(PerceptionException):
    """Raised when OCR operation fails."""

    def __init__(self, reason: str, **kwargs) -> None:
        """Initialize with OCR details."""
        super().__init__(
            f"OCR operation failed: {reason}",
            error_code="OCR_FAILED",
            context={"reason": reason, **kwargs},
        )


class ImageProcessingError(PerceptionException):
    """Raised when image processing operation fails."""

    def __init__(self, reason: str, image_path: str | None = None, **kwargs) -> None:
        """Initialize with image processing details."""
        message = "Image processing failed"
        if image_path:
            message += f" for image '{image_path}'"
        message += f": {reason}"

        super().__init__(
            message,
            error_code="IMAGE_PROCESSING_FAILED",
            context={"reason": reason, "image_path": image_path, **kwargs},
        )
