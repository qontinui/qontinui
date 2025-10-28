"""Screen capture abstraction."""

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ScreenCapturer:
    """Handles full screen capture operations.

    Provides clean interface to screen capture functionality,
    routing through the controller abstraction layer.
    """

    def capture(self) -> np.ndarray[Any, Any] | None:
        """Capture full screen.

        Returns:
            Screen image as BGR numpy array or None on error
        """
        try:
            from ......wrappers import get_controller

            controller = get_controller()
            screenshot = controller.capture.capture()

            # Convert PIL Image to numpy array (BGR for OpenCV)
            img_array = np.array(screenshot)

            # Convert RGB to BGR for OpenCV
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            logger.debug(f"Captured screen: {img_array.shape}")
            return img_array

        except Exception as e:
            logger.error(f"Error capturing screen: {e}", exc_info=True)
            return None
