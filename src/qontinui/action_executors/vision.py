"""Vision/image recognition action executor.

This module provides the VisionActionExecutor for handling image recognition
actions (FIND, VANISH) using template matching with mask support.
"""

import base64
import logging
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image as PILImage

from ..config import FindActionConfig, VanishActionConfig
from ..config.schema import Action
from ..exceptions import ActionExecutionError
from ..model.element import Pattern
from .base import ActionExecutorBase
from .registry import register_executor

logger = logging.getLogger(__name__)


@register_executor
class VisionActionExecutor(ActionExecutorBase):
    """Executor for vision/image recognition actions.

    Handles:
        - FIND: Locate image on screen, store location
        - VANISH: Wait for image to disappear

    Uses OpenCV template matching with configurable similarity thresholds.
    Results are stored in context.last_find_location for subsequent targeting.
    """

    def get_supported_action_types(self) -> list[str]:
        """Get list of vision action types this executor handles.

        Returns:
            List containing ["FIND", "VANISH"]
        """
        return ["FIND", "VANISH"]

    def execute(self, action: Action, typed_config: Any) -> bool:
        """Execute vision action with validated configuration.

        Args:
            action: Pydantic Action model
            typed_config: Type-specific config (FindActionConfig, VanishActionConfig)

        Returns:
            bool: True if action succeeded, False otherwise

        Raises:
            ActionExecutionError: If action execution fails critically
        """
        # File-based debug logging at entry point
        import os
        import tempfile
        from datetime import datetime

        debug_log_path = os.path.join(tempfile.gettempdir(), "qontinui_vision_executor_debug.log")

        try:
            with open(debug_log_path, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(f"[{timestamp}] VisionActionExecutor.execute() called\n")
                f.write(f"[{timestamp}]   Action ID: {action.id}\n")
                f.write(f"[{timestamp}]   Action type: {action.type}\n")
                f.write(f"[{timestamp}]   Typed config type: {type(typed_config).__name__}\n")
        except Exception:
            pass

        logger.debug(f"Executing vision action: {action.type}")

        # Route to appropriate handler
        if action.type == "FIND":
            return self._execute_find(action, typed_config)
        elif action.type == "VANISH":
            return self._execute_vanish(action, typed_config)
        else:
            raise ActionExecutionError(
                action_type=action.type, reason=f"Unknown vision action type: {action.type}"
            )

    def _execute_find(self, action: Action, typed_config: FindActionConfig) -> bool:
        """Execute FIND action - locate image and store location.

        Args:
            action: Pydantic Action model
            typed_config: Pre-validated FindActionConfig

        Returns:
            bool: True if image was found
        """
        # File-based debug logging
        import os
        import tempfile
        from datetime import datetime

        debug_log = os.path.join(tempfile.gettempdir(), "qontinui_event_emission.log")
        try:
            with open(debug_log, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(f"[{ts}] VisionActionExecutor._execute_find() ENTRY\n")
                f.write(f"[{ts}]   action.id={action.id}, action.type={action.type}\n")
                f.write(f"[{ts}]   target type: {type(typed_config.target).__name__}\n")
        except Exception:
            pass

        logger.debug(
            f"[FIND_DEBUG] _execute_find() ENTRY - action.id={action.id}, action.type={action.type}"
        )
        logger.debug(f"[FIND_DEBUG] typed_config.target type: {type(typed_config.target).__name__}")

        try:
            logger.debug("[FIND_DEBUG] Calling _get_target_location_from_typed()")

            # File-based debug logging before call
            try:
                with open(debug_log, "a", encoding="utf-8") as f:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}]   Calling _get_target_location_from_typed()...\n")
            except Exception:
                pass

            location = self._get_target_location_from_typed(typed_config.target)

            # File-based debug logging after call
            try:
                with open(debug_log, "a", encoding="utf-8") as f:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}]   _get_target_location_from_typed() returned: {location}\n")
            except Exception:
                pass

            logger.debug(f"[FIND_DEBUG] _get_target_location_from_typed() returned: {location}")

            if location:
                logger.info(f"FIND action succeeded: found at {location}")
                return True
            else:
                logger.error("[FIND_DEBUG] FIND action failed: target not found (location is None)")

                # File-based debug logging for failure
                try:
                    with open(debug_log, "a", encoding="utf-8") as f:
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        f.write(f"[{ts}]   FIND FAILED: location is None\n")
                except Exception:
                    pass

                return False

        except Exception as e:
            error_msg = f"FIND action failed: {e}"
            logger.error(f"[FIND_DEBUG] EXCEPTION in _execute_find: {error_msg}", exc_info=True)

            # File-based debug logging for exception
            try:
                with open(debug_log, "a", encoding="utf-8") as f:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}]   EXCEPTION in _execute_find: {error_msg}\n")
                    import traceback

                    f.write(f"[{ts}]   Traceback: {traceback.format_exc()}\n")
            except Exception:
                pass

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
        poll_interval = typed_config.poll_interval or 500  # Default 500ms

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

            logger.warning(
                f"VANISH action failed: element did not vanish within {timeout_seconds}s"
            )
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

        logger.debug(
            f"[FIND_DEBUG] _get_target_location_from_typed() ENTRY - target type: {type(target_config).__name__}"
        )

        # Handle different target types
        if isinstance(target_config, ImageTarget):
            # Find images on screen (supports multiple image_ids)
            image_ids = target_config.image_ids
            similarity = DEFAULT_SIMILARITY_THRESHOLD

            # Get similarity from search options if available
            if target_config.search_options and target_config.search_options.similarity is not None:
                similarity = target_config.search_options.similarity

            # Get search strategy from search options (default: FIRST)
            strategy = "FIRST"
            if target_config.search_options and target_config.search_options.strategy:
                strategy = target_config.search_options.strategy

            logger.debug(
                f"[FIND_DEBUG] ImageTarget: image_ids={image_ids}, similarity={similarity}, strategy={strategy}"
            )

            if image_ids:
                # Load all patterns
                logger.debug(
                    f"[FIND_DEBUG] Calling _load_patterns_from_config with {len(image_ids)} image IDs"
                )
                patterns = self._load_patterns_from_config(image_ids)
                logger.debug(
                    f"[FIND_DEBUG] _load_patterns_from_config returned {len(patterns) if patterns else 0} patterns"
                )

                if patterns:
                    logger.debug(
                        f"[FIND_DEBUG] Calling _find_patterns_on_screen with {len(patterns)} patterns"
                    )
                    result = self._find_patterns_on_screen(
                        patterns, similarity, image_ids, strategy
                    )
                    logger.debug(f"[FIND_DEBUG] _find_patterns_on_screen returned: {result}")
                    return result
                else:
                    logger.error(
                        f"[FIND_DEBUG] Could not load any patterns for image IDs: {image_ids}"
                    )
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

    def _load_patterns_from_config(self, image_ids: list[str]) -> list[Pattern]:
        """Load multiple Patterns from config.

        This method loads all patterns for a list of image IDs. If a StateImage has
        multiple pattern variations, all are included in the search.

        Args:
            image_ids: List of image IDs to load

        Returns:
            List of Pattern objects (may be empty if none found)
        """
        logger.debug(f"[FIND_DEBUG] _load_patterns_from_config() ENTRY - image_ids={image_ids}")
        patterns = []

        for image_id in image_ids:
            try:
                logger.debug(f"[FIND_DEBUG] Processing image_id: {image_id}")
                # For state images, load ALL patterns (not just first)
                if image_id.startswith("stateimage-"):
                    logger.debug(f"[FIND_DEBUG] State image detected: {image_id}")
                    for state in self.context.config.states:
                        for state_image in state.identifying_images:
                            if state_image.id == image_id:
                                logger.debug(
                                    f"[FIND_DEBUG] Found state image {image_id} with {len(state_image.patterns)} patterns"
                                )
                                # Load ALL patterns for this state image
                                for i, pattern_config in enumerate(state_image.patterns):
                                    logger.debug(
                                        f"[FIND_DEBUG] Loading pattern {i+1}/{len(state_image.patterns)} from state image"
                                    )
                                    pattern = self._build_pattern_from_config(
                                        pattern_config, image_id
                                    )
                                    if pattern:
                                        logger.debug(
                                            f"[FIND_DEBUG] Pattern loaded successfully: {pattern.id}"
                                        )
                                        patterns.append(pattern)
                                    else:
                                        logger.error(
                                            "[FIND_DEBUG] Pattern failed to load (returned None)"
                                        )
                                break
                else:
                    # Single pattern lookup
                    logger.debug(f"[FIND_DEBUG] Direct image lookup for: {image_id}")
                    pattern = self._load_pattern_from_config(image_id)
                    if pattern:
                        logger.debug(f"[FIND_DEBUG] Pattern loaded successfully: {pattern.id}")
                        patterns.append(pattern)
                    else:
                        logger.error("[FIND_DEBUG] Pattern failed to load (returned None)")
            except Exception as e:
                logger.error(
                    f"[FIND_DEBUG] EXCEPTION loading pattern for image {image_id}: {e}",
                    exc_info=True,
                )
                continue

        logger.debug(
            f"[FIND_DEBUG] _load_patterns_from_config() RETURN - Loaded {len(patterns)} patterns from {len(image_ids)} image IDs"
        )
        return patterns

    def _load_pattern_from_config(self, image_id: str) -> Pattern | None:
        """Load Pattern with mask from config.

        This method looks up the image in state images and patterns to extract
        the complete pattern data including mask information.

        Args:
            image_id: Image ID to look up (can be state image ID or underlying image ID)

        Returns:
            Pattern object with pixel_data and mask, or None if not found
        """
        try:
            # Check if this is a state image ID (starts with "stateimage-")
            if image_id.startswith("stateimage-"):
                logger.debug(f"Looking up state image: {image_id}")
                # Find the state image by ID across all states
                for state in self.context.config.states:
                    for state_image in state.identifying_images:
                        if state_image.id == image_id:
                            logger.debug(
                                f"Found state image {image_id} with {len(state_image.patterns)} patterns"
                            )
                            # Use the first pattern (state images typically have one pattern)
                            if state_image.patterns:
                                pattern_config = state_image.patterns[0]
                                return self._build_pattern_from_config(pattern_config, image_id)
                            else:
                                logger.warning(f"State image {image_id} has no patterns")
                                return None
            else:
                # It's a direct image ID, search for pattern that references it
                logger.debug(f"Looking up image by pattern reference: {image_id}")
                for state in self.context.config.states:
                    for state_image in state.identifying_images:
                        for pattern_config in state_image.patterns:
                            if pattern_config.image_id == image_id:
                                # Found the pattern config, now build a Pattern object
                                return self._build_pattern_from_config(pattern_config, image_id)

            # If not found in state images, try to load just the image without mask
            image_asset = self.context.config.image_map.get(image_id)
            if image_asset and image_asset.file_path:
                logger.debug(f"Loading image without mask from image_map: {image_id}")
                return self._build_pattern_from_file(image_asset.file_path, image_id)

            logger.warning(f"Image {image_id} not found in state images or image_map")
            return None

        except Exception as e:
            logger.error(f"Error loading pattern for image {image_id}: {e}", exc_info=True)
            return None

    def _build_pattern_from_config(self, pattern_config: Any, image_id: str) -> Pattern | None:
        """Build Pattern object from config with mask.

        Args:
            pattern_config: Pattern configuration from state image
            image_id: Image ID for this pattern

        Returns:
            Pattern object or None if loading fails
        """
        logger.debug(
            f"[FIND_DEBUG] _build_pattern_from_config() ENTRY - image_id={image_id}, pattern_config.image_id={pattern_config.image_id}"
        )

        try:
            # Load the base image
            image_asset = self.context.config.image_map.get(pattern_config.image_id)
            if not image_asset or not image_asset.file_path:
                logger.error(
                    f"[FIND_DEBUG] Image asset not found for pattern: {pattern_config.image_id}"
                )
                return None

            logger.debug(f"[FIND_DEBUG] Image asset found, file_path={image_asset.file_path}")

            # Load image as numpy array
            logger.debug(f"[FIND_DEBUG] Calling cv2.imread({image_asset.file_path})")
            pixel_data = cv2.imread(image_asset.file_path)

            if pixel_data is None:
                logger.error(
                    f"[FIND_DEBUG] cv2.imread() returned None for file: {image_asset.file_path}"
                )
                import os

                logger.error(f"[FIND_DEBUG] File exists: {os.path.exists(image_asset.file_path)}")
                if os.path.exists(image_asset.file_path):
                    logger.error(
                        f"[FIND_DEBUG] File size: {os.path.getsize(image_asset.file_path)} bytes"
                    )
                return None

            logger.debug(
                f"[FIND_DEBUG] cv2.imread() succeeded - pixel_data shape: {pixel_data.shape}"
            )

            # Decode mask if present
            if pattern_config.mask:
                logger.debug("[FIND_DEBUG] Pattern has mask, decoding...")
                mask = self._decode_mask(pattern_config.mask, pixel_data.shape[:2])
            else:
                logger.debug("[FIND_DEBUG] Pattern has no mask")
                mask = None

            # Create Pattern object
            # Use pattern_config.name if it's not empty, otherwise fall back to image_id
            pattern_name = (
                pattern_config.name
                if pattern_config.name and pattern_config.name.strip()
                else image_id
            )

            pattern = Pattern(
                id=pattern_config.id or image_id,
                name=pattern_name,
                pixel_data=pixel_data,
                mask=mask if mask is not None else np.ones(pixel_data.shape[:2], dtype=np.float32),
            )

            logger.debug(
                f"[FIND_DEBUG] Pattern created successfully - id={pattern.id}, name={pattern.name}, has_mask={mask is not None}, pixel_data_is_None={pattern.pixel_data is None}"
            )
            return pattern

        except Exception as e:
            logger.error(
                f"[FIND_DEBUG] EXCEPTION in _build_pattern_from_config: {e}", exc_info=True
            )
            return None

    def _build_pattern_from_file(self, file_path: str, image_id: str) -> Pattern | None:
        """Build Pattern object from image file without mask.

        Args:
            file_path: Path to image file
            image_id: Image ID for the pattern

        Returns:
            Pattern object or None if loading fails
        """
        logger.debug(
            f"[FIND_DEBUG] _build_pattern_from_file() ENTRY - file_path={file_path}, image_id={image_id}"
        )

        try:
            logger.debug(f"[FIND_DEBUG] Calling cv2.imread({file_path})")
            pixel_data = cv2.imread(file_path)

            if pixel_data is None:
                logger.error(f"[FIND_DEBUG] cv2.imread() returned None for file: {file_path}")
                import os

                logger.error(f"[FIND_DEBUG] File exists: {os.path.exists(file_path)}")
                if os.path.exists(file_path):
                    logger.error(f"[FIND_DEBUG] File size: {os.path.getsize(file_path)} bytes")
                return None

            logger.debug(
                f"[FIND_DEBUG] cv2.imread() succeeded - pixel_data shape: {pixel_data.shape}"
            )

            # Create full mask (no masking)
            mask = np.ones(pixel_data.shape[:2], dtype=np.float32)

            pattern = Pattern(id=image_id, name=image_id, pixel_data=pixel_data, mask=mask)

            logger.debug(
                f"[FIND_DEBUG] Pattern created successfully - id={pattern.id}, pixel_data_is_None={pattern.pixel_data is None}"
            )
            return pattern

        except Exception as e:
            logger.error(f"[FIND_DEBUG] EXCEPTION in _build_pattern_from_file: {e}", exc_info=True)
            return None

    def _decode_mask(self, mask_data: str, shape: tuple[int, int]) -> np.ndarray | None:
        """Decode base64 mask image to numpy array.

        Args:
            mask_data: Base64-encoded mask image (data:image/png;base64,...)
            shape: Expected shape (height, width)

        Returns:
            Mask as numpy array (0.0-1.0 float) or None if decoding fails
        """
        try:
            # Strip data URI prefix if present
            if mask_data.startswith("data:image"):
                mask_data = mask_data.split(",", 1)[1]

            # Decode base64
            mask_bytes = base64.b64decode(mask_data)

            # Load as PIL Image
            mask_img = PILImage.open(BytesIO(mask_bytes))

            # Convert to grayscale if needed
            if mask_img.mode != "L":
                mask_img = mask_img.convert("L")

            # Convert to numpy array (0-255)
            mask_array = np.array(mask_img, dtype=np.uint8)

            # Resize if shape doesn't match
            if mask_array.shape != shape:
                logger.warning(
                    f"Mask shape {mask_array.shape} doesn't match image shape {shape}, resizing"
                )
                mask_img_resized = mask_img.resize(
                    (shape[1], shape[0]), PILImage.Resampling.LANCZOS
                )
                mask_array = np.array(mask_img_resized, dtype=np.uint8)

            # Normalize to 0.0-1.0 float
            mask_normalized = mask_array.astype(np.float32) / 255.0

            logger.debug(
                f"Decoded mask: shape={mask_normalized.shape}, non-zero pixels={np.count_nonzero(mask_normalized > 0.5)}"
            )
            return mask_normalized

        except Exception as e:
            logger.error(f"Error decoding mask: {e}")
            return None

    def _find_patterns_on_screen(
        self,
        patterns: list[Pattern],
        threshold: float,
        image_ids: list[str],
        strategy: str = "FIRST",
    ) -> tuple[int, int] | None:
        """Find multiple patterns on screen with strategy-based selection.

        Uses async pattern matching for performance when multiple patterns are present.

        Args:
            patterns: List of patterns to find
            threshold: Similarity threshold (0.0-1.0)
            image_ids: List of image IDs (for logging and EACH strategy)
            strategy: Search strategy (FIRST, BEST, ALL, EACH)

        Returns:
            Tuple of (x, y) coordinates if found, None otherwise
        """
        from ..actions.find import FindAction, FindOptions

        # Single pattern: use synchronous path (backward compatible)
        if len(patterns) == 1:
            return self._find_pattern_on_screen(
                patterns[0], threshold, image_ids[0] if image_ids else None
            )

        # Multiple patterns: use async execution for performance
        logger.debug(f"Finding {len(patterns)} patterns with strategy {strategy}")

        import asyncio

        # Run async find in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def find_all_patterns():
            """Find all patterns using async parallelism."""

            action = FindAction()
            tasks = []

            for pattern in patterns:
                # Create a task for each pattern
                task = asyncio.create_task(
                    asyncio.to_thread(
                        action.find,
                        pattern=pattern,
                        options=FindOptions(
                            similarity=threshold,
                            find_all=False,
                        ),
                    )
                )
                tasks.append((pattern, task))

            # For FIRST strategy, use as_completed to return immediately
            if strategy == "FIRST":
                for pattern, task in tasks:
                    try:
                        result = await task
                        if result.found and result.best_match:
                            # Cancel remaining tasks
                            for _, t in tasks:
                                if not t.done():
                                    t.cancel()
                            return result
                    except Exception as e:
                        logger.warning(f"Pattern {pattern.id} search failed: {e}")
                        continue
                return None

            # For BEST/ALL/EACH, gather all results
            results = []
            for pattern, task in tasks:
                try:
                    result = await task
                    if result.found and result.best_match:
                        results.append((pattern, result))
                except Exception as e:
                    logger.warning(f"Pattern {pattern.id} search failed: {e}")
                    continue

            return results

        # Execute async search
        result = loop.run_until_complete(find_all_patterns())

        # Apply strategy to results
        if strategy == "FIRST":
            # Result is already the first match or None
            if result and result.found and result.best_match:
                match = result.best_match
                center_x = match.target.region.x + match.target.region.width // 2
                center_y = match.target.region.y + match.target.region.height // 2
                location = (center_x, center_y)
                self.context.update_last_find_location(location)
                logger.debug(f"FIRST strategy: Found at {location}")
                return location
            return None

        elif strategy == "BEST":
            # Find the match with highest confidence
            if not result:
                return None

            best_result = max(result, key=lambda r: r[1].best_match.confidence)
            match = best_result[1].best_match
            center_x = match.target.region.x + match.target.region.width // 2
            center_y = match.target.region.y + match.target.region.height // 2
            location = (center_x, center_y)
            self.context.update_last_find_location(location)
            logger.debug(f"BEST strategy: Found at {location} with confidence {match.confidence}")
            return location

        elif strategy == "ALL":
            # Return first match (for now - in future, could return all matches)
            # Note: ExecutionContext.last_find_location only supports single location
            if not result:
                return None

            # Use the best match among all
            best_result = max(result, key=lambda r: r[1].best_match.confidence)
            match = best_result[1].best_match
            center_x = match.target.region.x + match.target.region.width // 2
            center_y = match.target.region.y + match.target.region.height // 2
            location = (center_x, center_y)
            self.context.update_last_find_location(location)
            logger.debug(f"ALL strategy: Found {len(result)} matches, using best at {location}")
            return location

        elif strategy == "EACH":
            # Return best match per StateImage (grouped by image_id)
            # For now, return first match (similar to FIRST)
            if not result:
                return None

            # Use the first match
            match = result[0][1].best_match
            center_x = match.target.region.x + match.target.region.width // 2
            center_y = match.target.region.y + match.target.region.height // 2
            location = (center_x, center_y)
            self.context.update_last_find_location(location)
            logger.debug(f"EACH strategy: Found {len(result)} matches, using first at {location}")
            return location

        return None

    def _find_pattern_on_screen(
        self,
        pattern: Pattern,
        threshold: float,
        image_id: str | None = None,
    ) -> tuple[int, int] | None:
        """Find pattern on screen using unified FindAction.

        Args:
            pattern: Pattern to find (includes image + mask)
            threshold: Similarity threshold (0.0-1.0)
            image_id: Optional image identifier for logging

        Returns:
            Tuple of (x, y) coordinates if found, None otherwise
        """
        from ..actions.find import FindAction, FindOptions

        logger.debug(
            f"[FIND_DEBUG] _find_pattern_on_screen() ENTRY - pattern.id={pattern.id}, pattern.name={pattern.name}, threshold={threshold}"
        )
        logger.debug(f"[FIND_DEBUG] Pattern pixel_data is None: {pattern.pixel_data is None}")
        if pattern.pixel_data is not None:
            logger.debug(f"[FIND_DEBUG] Pattern pixel_data shape: {pattern.pixel_data.shape}")

        # Use FindAction (single entry point for all finding)
        logger.debug("[FIND_DEBUG] Creating FindAction and calling find()")
        action = FindAction()
        result = action.find(
            pattern=pattern,
            options=FindOptions(
                similarity=threshold,
                find_all=False,
            ),
        )

        logger.debug(
            f"[FIND_DEBUG] FindAction.find() returned - found={result.found}, best_match={result.best_match is not None}"
        )

        # Return location if found
        if result.found and result.best_match:
            match = result.best_match
            center_x = match.target.region.x + match.target.region.width // 2
            center_y = match.target.region.y + match.target.region.height // 2
            location = (center_x, center_y)

            # CRITICAL: Update last_find_location so MOUSE_MOVE/CLICK can use it
            self.context.update_last_find_location(location)
            logger.debug(f"[FIND_DEBUG] Updated last_find_location to {location}")

            return location

        logger.debug("[FIND_DEBUG] Pattern not found, returning None")
        return None
