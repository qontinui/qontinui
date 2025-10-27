"""Image processing exception.

Exception thrown when image processing operations fail.
"""

from .qontinui_runtime_exception import QontinuiRuntimeException


class ImageProcessingError(QontinuiRuntimeException):
    """Exception thrown when image processing operations fail.

    Raised during image template matching, feature detection,
    or other computer vision operations.
    """

    def __init__(
        self,
        message: str = "Image processing failed",
        cause: Exception | None = None,
        image_path: str | None = None,
    ):
        """Initialize image processing exception.

        Args:
            message: Error message
            cause: Underlying exception that caused this error
            image_path: Path to image that failed processing (if applicable)
        """
        super().__init__(message, cause)
        self.image_path = image_path
