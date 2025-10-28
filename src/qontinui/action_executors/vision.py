"""Vision/image recognition action executor.

This module provides the VisionActionExecutor for handling image recognition
actions (FIND, EXISTS, VANISH) using OpenCV template matching.
"""

import logging
from typing import Any

import cv2
import numpy as np

from ..config import ExistsActionConfig, FindActionConfig, VanishActionConfig
from ..config.schema import Action
from ..exceptions import ActionExecutionError, ImageProcessingError
from .base import ActionExecutorBase, ExecutionContext
from .registry import register_executor

logger = logging.getLogger(__name__)


@register_executor
class VisionActionExecutor(ActionExecutorBase):
    """Executor for vision/image recognition actions.

    Handles:
        - FIND: Locate image on screen, store location
        - EXISTS: Check if image exists (boolean result)
        - VANISH: Wait for image to disappear

    Uses OpenCV template matching with configurable similarity thresholds.
    Results are stored in context.last_find_location for subsequent targeting.
    """

    def get_supported_action_types(self) -> list[str]:
        """Get list of vision action types this executor handles.

        Returns:
            List containing ["FIND", "EXISTS", "VANISH"]
        """
        return ["FIND", "EXISTS", "VANISH"]

    def execute(self, action: Action, typed_config: Any) -> bool:
        """Execute vision action with validated configuration.

        Args:
            action: Pydantic Action model
            typed_config: Type-specific config (FindActionConfig, ExistsActionConfig, VanishActionConfig)

        Returns:
            bool: True if action succeeded, False otherwise

        Raises:
            ActionExecutionError: If action execution fails critically
        """
        logger.debug(f"Executing vision action: {action.type}")

        # Route to appropriate handler
        if action.type == "FIND":
            return self._execute_find(action, typed_config)
        elif action.type == "EXISTS":
            return self._execute_exists(action, typed_config)
        elif action.type == "VANISH":
            return self._execute_vanish(action, typed_config)
        else:
            raise ActionExecutionError(
                action_type=action.type,
                reason=f"Unknown vision action type: {action.type}"
            )

    def _execute_find(self, action: Action, typed_config: FindActionConfig) -> bool:
        """Execute FIND action - locate image and store location.

        Args:
            action: Pydantic Action model
            typed_config: Pre-validated FindActionConfig

        Returns:
            bool: True if image was found
        """
        logger.debug("Executing FIND action")

        try:
            location = self._get_target_location_from_typed(typed_config.target)

            if location:
                logger.info(f"FIND action succeeded: found at {location}")
                return True
            else:
                logger.warning("FIND action failed: target not found")
                return False

        except Exception as e:
            error_msg = f"FIND action failed: {e}"
            logger.error(error_msg)
            self._emit_action_failure(action, error_msg)
            return False

    def _execute_exists(self, action: Action, typed_config: ExistsActionConfig) -> bool:
        """Execute EXISTS action - check if image exists.

        Args:
            action: Pydantic Action model
            typed_config: Pre-validated ExistsActionConfig

        Returns:
            bool: True if image exists, False otherwise
        """
        logger.debug("Executing EXISTS action")

        try:
            location = self._get_target_location_from_typed(typed_config.target)
            exists = location is not None

            logger.info(f"EXISTS action: element exists = {exists}")

            # Store result in variable if specified
            if typed_config.output_variable:
                self.context.variable_context.set(
                    typed_config.output_variable,
                    exists,
                    "local"
                )
                logger.debug(f"Stored result in variable: {typed_config.output_variable} = {exists}")

            return exists

        except Exception as e:
            error_msg = f"EXISTS action failed: {e}"
            logger.error(error_msg)
            self._emit_action_failure(action, error_msg)
            return False

    def _execute_vanish(self, action: Action, typed_config: VanishActionConfig) -> bool:
        """Execute VANISH action - wait for image to disappear.

        Args:
            action: Pydantic Action model
            typed_config: Pre-validated VanishActionConfig

        Returns:
            bool: True if element vanished within timeout
        """
        logger.debug("Executing VANISH action")

        # Get timeout from config or action
        max_wait_time = typed_config.max_wait_time or 30000  # Default 30 seconds
        poll_interval = typed_config.poll_interval or 500     # Default 500ms

        timeout_seconds = max_wait_time / 1000.0
        poll_seconds = poll_interval / 1000.0

        logger.debug(f"VANISH timeout: {timeout_seconds}s, poll interval: {poll_seconds}s")

        try:
            start_time = self.context.time.now()
            elapsed = 0.0

            while elapsed < timeout_seconds:
                location = self._get_target_location_from_typed(typed_config.target)

                if location is None:
                    logger.info(f"VANISH action succeeded: element vanished after {elapsed:.2f}s")
                    return True

                # Wait before next poll
                self.context.time.wait(poll_seconds)
                elapsed = (self.context.time.now() - start_time).total_seconds()

            logger.warning(f"VANISH action failed: element did not vanish within {timeout_seconds}s")
            return False

        except Exception as e:
            error_msg = f"VANISH action failed: {e}"
            logger.error(error_msg)
            self._emit_action_failure(action, error_msg)
            return False

    def _get_target_location_from_typed(self, target_config: Any) -> tuple[int, int] | None:
        """Get target location from typed TargetConfig.

        This method handles different target types and delegates to image recognition
        for ImageTarget types.

        Args:
            target_config: TargetConfig union type (ImageTarget, RegionTarget, CoordinatesTarget)

        Returns:
            Tuple of (x, y) coordinates or None if not found
        """
        from ..config import CoordinatesTarget, ImageTarget, RegionTarget
        from ..json_executor.constants import DEFAULT_SIMILARITY_THRESHOLD

        logger.debug(f"Getting location from typed target: {type(target_config).__name__}")

        # Handle different target types
        if isinstance(target_config, ImageTarget):
            # Find image on screen
            image_id = target_config.image_id
            similarity = DEFAULT_SIMILARITY_THRESHOLD

            # Get similarity from search options if available
            if target_config.search_options and target_config.search_options.similarity is not None:
                similarity = target_config.search_options.similarity

            logger.debug(f"Finding image {image_id} with similarity {similarity}")

            if image_id:
                image = self.context.config.image_map.get(image_id)
                if image and image.file_path:
                    return self._find_image_on_screen(image.file_path, similarity)
                else:
                    if not image:
                        logger.error(f"Image ID not found in image_map: {image_id}")
                    else:
                        logger.error(f"Image file_path is None for image: {image_id}")
                    return None

        elif isinstance(target_config, CoordinatesTarget):
            coords = target_config.coordinates
            logger.debug(f"Using coordinates: ({coords.x}, {coords.y})")
            return (coords.x, coords.y)

        elif isinstance(target_config, RegionTarget):
            region = target_config.region
            # Return center of region
            x = region.x + region.width // 2
            y = region.y + region.height // 2
            logger.debug(f"Using region center: ({x}, {y})")
            return (x, y)

        elif isinstance(target_config, str) and target_config == "Last Find Result":
            # Handle string target for backward compatibility
            if self.context.last_find_location:
                logger.debug(f"Using Last Find Result: {self.context.last_find_location}")
                return self.context.last_find_location
            else:
                logger.error("Last Find Result requested but no previous find result available")
                return None

        logger.error(f"Unsupported target type: {type(target_config)}")
        return None

    def _find_image_on_screen(
        self,
        image_path: str,
        threshold: float
    ) -> tuple[int, int] | None:
        """Find image on screen using template matching.

        Args:
            image_path: Path to template image file
            threshold: Similarity threshold (0.0 to 1.0)

        Returns:
            Tuple of (x, y) coordinates of image center, or None if not found

        Raises:
            ImageProcessingError: If image loading or processing fails
        """
        try:
            # Take screenshot using wrapper
            screenshot = self.context.screen.capture()
            # Screen.capture() returns np.ndarray, but convert to ensure proper format
            screenshot_np = np.array(screenshot) if not isinstance(screenshot, np.ndarray) else screenshot
            screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)

            # Load template image
            template = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                error_msg = f"Failed to load template image: {image_path}"
                logger.error(error_msg)
                self.context.emit_image_recognition_event({
                    "image_path": image_path,
                    "template_size": "unknown",
                    "screenshot_size": f"{screenshot_gray.shape[1]}x{screenshot_gray.shape[0]}",
                    "threshold": threshold,
                    "confidence": 0,
                    "found": False,
                    "error": error_msg,
                })
                raise ImageProcessingError(reason=error_msg, image_path=image_path)

            template_size = f"{template.shape[1]}x{template.shape[0]}"
            screenshot_size = f"{screenshot_gray.shape[1]}x{screenshot_gray.shape[0]}"

            # Debug info
            logger.debug(f"Image Recognition - Template: {image_path}")
            logger.debug(f"Template size: {template_size}, Screenshot size: {screenshot_size}")
            logger.debug(f"Threshold: {threshold:.2f}")

            # Template matching
            result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            logger.debug(f"Confidence: {max_val:.4f} (need >= {threshold:.2f})")

            if max_val >= threshold:
                # Return center of found image
                h, w = template.shape
                center_x = max_loc[0] + w // 2
                center_y = max_loc[1] + h // 2
                location_str = f"({center_x}, {center_y})"
                logger.info(f"Image found at {location_str}")

                # Store as last find location for "Last Find Result" targeting
                self.context.update_last_find_location((center_x, center_y))

                # Emit success event
                self.context.emit_image_recognition_event({
                    "image_path": image_path,
                    "template_size": template_size,
                    "screenshot_size": screenshot_size,
                    "threshold": threshold,
                    "confidence": max_val,
                    "found": True,
                    "location": location_str,
                })

                return (center_x, center_y)
            else:
                # Calculate how close we were
                gap = threshold - max_val
                percent_off = (gap / threshold) * 100
                best_match_str = f"({max_loc[0]}, {max_loc[1]})"
                logger.debug(f"Image not found (missed by {gap:.4f} / {percent_off:.1f}%)")
                logger.debug(f"Best match location: {best_match_str}")

                # Emit failure event
                self.context.emit_image_recognition_event({
                    "image_path": image_path,
                    "template_size": template_size,
                    "screenshot_size": screenshot_size,
                    "threshold": threshold,
                    "confidence": max_val,
                    "found": False,
                    "gap": gap,
                    "percent_off": percent_off,
                    "best_match_location": best_match_str,
                })

                return None

        except cv2.error as e:
            error_msg = f"OpenCV error during image recognition: {e}"
            logger.error(error_msg)
            self.context.emit_image_recognition_event({
                "image_path": image_path,
                "template_size": "unknown",
                "screenshot_size": "unknown",
                "threshold": threshold,
                "confidence": 0,
                "found": False,
                "error": error_msg,
            })
            raise ImageProcessingError(reason=error_msg, image_path=image_path) from e

        except Exception as e:
            error_msg = f"Error finding image: {e}"
            logger.error(error_msg)
            self.context.emit_image_recognition_event({
                "image_path": image_path,
                "template_size": "unknown",
                "screenshot_size": "unknown",
                "threshold": threshold,
                "confidence": 0,
                "found": False,
                "error": error_msg,
            })
            raise ImageProcessingError(reason=error_msg, image_path=image_path) from e
