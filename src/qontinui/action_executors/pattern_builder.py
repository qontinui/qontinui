"""Pattern building from configuration and files.

Handles loading images from disk and decoding masks to create Pattern objects.
"""

import base64
import logging
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

from ..model.element import Pattern

logger = logging.getLogger(__name__)


class PatternBuilder:
    """Builds Pattern objects from configuration or file paths.

    Responsibilities:
    - Load images from disk using cv2
    - Decode base64-encoded masks
    - Create Pattern objects with pixel data and masks
    """

    def build_from_config(
        self, pattern_config: dict, image_id: str, image_assets_dir: Path
    ) -> Pattern | None:
        """Build Pattern from configuration with optional mask.

        Args:
            pattern_config: Dict with 'path' and optional 'mask' keys
            image_id: Identifier for this pattern
            image_assets_dir: Directory containing image assets

        Returns:
            Pattern object or None if image loading fails
        """
        image_path = pattern_config.get("path")
        if not image_path:
            logger.error(f"No path in pattern config for image_id={image_id}")
            return None

        # Resolve absolute path
        full_path = image_assets_dir / image_path

        # Load image
        pixel_data = self._load_image_file(full_path)
        if pixel_data is None:
            return None

        # Decode mask if present
        mask_data = pattern_config.get("mask")
        if mask_data:
            mask = self._decode_mask(mask_data, pixel_data.shape)
            if mask is None:
                logger.warning(
                    f"Failed to decode mask for {image_id}, using default mask"
                )
                mask = np.ones(pixel_data.shape[:2], dtype=np.uint8) * 255
        else:
            # Default: full white mask (all pixels considered)
            mask = np.ones(pixel_data.shape[:2], dtype=np.uint8) * 255

        # Create pattern
        pattern_name = pattern_config.get("name", image_id)
        return Pattern(id=image_id, pixel_data=pixel_data, mask=mask, name=pattern_name)

    def build_from_file(self, file_path: Path, image_id: str) -> Pattern | None:
        """Build Pattern from file path without mask.

        Args:
            file_path: Path to image file
            image_id: Identifier for this pattern

        Returns:
            Pattern object with default mask or None if loading fails
        """
        pixel_data = self._load_image_file(file_path)
        if pixel_data is None:
            return None

        # Default mask (all pixels considered)
        mask = np.ones(pixel_data.shape[:2], dtype=np.uint8) * 255

        return Pattern(id=image_id, pixel_data=pixel_data, mask=mask, name=image_id)

    def _load_image_file(self, file_path: Path) -> np.ndarray | None:
        """Load image file using cv2.

        Args:
            file_path: Path to image file

        Returns:
            Numpy array with image data or None if loading fails
        """
        if not file_path.exists():
            logger.error(f"Image file not found: {file_path}")
            return None

        try:
            pixel_data = cv2.imread(str(file_path))
            if pixel_data is None:
                logger.error(f"Failed to load image: {file_path}")
                return None

            logger.debug(f"Loaded image: {file_path} (shape={pixel_data.shape})")
            return pixel_data

        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            return None

    def _decode_mask(self, mask_data: str, image_shape: tuple) -> np.ndarray | None:
        """Decode base64-encoded mask to numpy array.

        Args:
            mask_data: Base64-encoded mask string
            image_shape: Shape of the image (height, width, channels)

        Returns:
            Numpy array mask (height x width) or None if decoding fails
        """
        try:
            # Decode base64
            mask_bytes = base64.b64decode(mask_data)

            # Load as PIL image
            mask_pil = PILImage.open(BytesIO(mask_bytes))

            # Convert to numpy array
            mask_array = np.array(mask_pil)

            # Ensure single channel (grayscale)
            if mask_array.ndim == 3:
                mask_array = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)

            # Normalize to 0-255 range
            if mask_array.dtype != np.uint8:
                mask_array = (mask_array / mask_array.max() * 255).astype(np.uint8)

            # Verify mask dimensions match image
            expected_height, expected_width = image_shape[:2]
            if mask_array.shape != (expected_height, expected_width):
                logger.warning(
                    f"Mask shape {mask_array.shape} != image shape "
                    f"({expected_height}, {expected_width})"
                )

            logger.debug(
                f"Decoded mask: shape={mask_array.shape}, dtype={mask_array.dtype}"
            )
            return mask_array

        except Exception as e:
            logger.error(f"Failed to decode mask: {e}")
            return None
