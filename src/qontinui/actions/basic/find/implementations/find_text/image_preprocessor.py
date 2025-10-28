"""Image preprocessing for OCR.

Handles image preprocessing operations to improve OCR accuracy.
"""

import logging
from typing import Any

import numpy as np

from ...options.text_find_options import TextPreprocessing

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocesses images for OCR.

    Applies various image transformations to improve text recognition.
    """

    def __init__(self, preprocessing: list[TextPreprocessing], scale_factor: float = 1.0) -> None:
        """Initialize image preprocessor.

        Args:
            preprocessing: List of preprocessing operations to apply
            scale_factor: Image scaling factor (>1.0 upscales, <1.0 downscales)
        """
        self.preprocessing = preprocessing
        self.scale_factor = scale_factor

    def preprocess(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess image for OCR.

        Args:
            image: Input image

        Returns:
            Preprocessed image
        """
        import cv2

        result = image.copy()

        # Apply preprocessing operations in sequence
        for operation in self.preprocessing:
            result = self._apply_operation(result, operation)

        # Scale image if needed
        if self.scale_factor != 1.0:
            result = self._scale_image(result, self.scale_factor)

        return result

    def _apply_operation(
        self, image: np.ndarray[Any, Any], operation: TextPreprocessing
    ) -> np.ndarray[Any, Any]:
        """Apply a single preprocessing operation.

        Args:
            image: Input image
            operation: Preprocessing operation to apply

        Returns:
            Processed image
        """
        import cv2

        if operation == TextPreprocessing.GRAYSCALE:
            return self._to_grayscale(image)

        elif operation == TextPreprocessing.BINARIZE:
            return self._binarize(image)

        elif operation == TextPreprocessing.DENOISE:
            return self._denoise(image)

        elif operation == TextPreprocessing.ENHANCE:
            return self._enhance_contrast(image)

        elif operation == TextPreprocessing.DESKEW:
            logger.warning("Deskew preprocessing not yet implemented")
            return image

        return image

    def _to_grayscale(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Convert image to grayscale.

        Args:
            image: Input image

        Returns:
            Grayscale image
        """
        import cv2

        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _binarize(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Binarize image (convert to black and white).

        Args:
            image: Input image

        Returns:
            Binarized image
        """
        import cv2

        # Convert to grayscale first if needed
        gray = self._to_grayscale(image)

        # Apply Otsu's thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def _denoise(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Remove noise from image.

        Args:
            image: Input image

        Returns:
            Denoised image
        """
        import cv2

        if len(image.shape) == 2:
            # Grayscale - use median blur
            return cv2.medianBlur(image, 3)
        else:
            # Color - use bilateral filter
            return cv2.bilateralFilter(image, 9, 75, 75)

    def _enhance_contrast(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Enhance image contrast.

        Args:
            image: Input image

        Returns:
            Contrast-enhanced image
        """
        import cv2

        if len(image.shape) == 2:
            # Grayscale - apply histogram equalization
            return cv2.equalizeHist(image)
        else:
            # Color - enhance L channel in LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            l_channel = cv2.equalizeHist(l_channel)
            return cv2.cvtColor(cv2.merge([l_channel, a, b]), cv2.COLOR_LAB2BGR)

    def _scale_image(
        self, image: np.ndarray[Any, Any], scale_factor: float
    ) -> np.ndarray[Any, Any]:
        """Scale image by a factor.

        Args:
            image: Input image
            scale_factor: Scaling factor

        Returns:
            Scaled image
        """
        import cv2

        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
