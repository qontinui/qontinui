"""Mouse action executor for all mouse-related actions.

This module provides the MouseActionExecutor class that handles all mouse operations
including movement, clicking, dragging, and scrolling.

# Multi-Monitor Coordinate System
# ===============================
#
# When targeting FIND results on multi-monitor setups, coordinate translation is critical.
#
# ## The Problem
#
# - FIND captures the ENTIRE virtual desktop (all monitors combined)
# - Match coordinates are relative to the virtual desktop screenshot origin
# - pyautogui needs absolute virtual desktop coordinates
#
# ## The Solution
#
# The Rust runner calculates the "virtual desktop origin" (minimum X, minimum Y across
# all monitors) and passes it to Python as `monitor_offset_x` and `monitor_offset_y`
# via the execution context.
#
# When resolving LastFindResultTarget, ResultIndexTarget, etc., this offset is added
# to the FIND result coordinates:
#
#     final_x = match.x + context.monitor_offset_x
#     final_y = match.y + context.monitor_offset_y
#
# ## Example
#
# Monitor layout:
#     Left: (-1920, 702), 1920x1080
#     Primary: (0, 0), 3840x2160
#     Right: (3840, 702), 1920x1080
#
# Virtual desktop origin: (-1920, 0)  # min X=-1920, min Y=0
#
# FIND result on left monitor: (65, 1372)
# After offset: (65 + -1920, 1372 + 0) = (-1855, 1372)
# pyautogui moves mouse to (-1855, 1372) -> lands on left monitor correctly!
#
# ## Key Files in the Coordinate Chain
#
# 1. mcp_api.rs (Rust) - Calculates virtual desktop origin from Tauri monitors
# 2. mouse.py (Python) - Applies offset when resolving target locations
# 3. mss_capture.py (Python) - Captures virtual desktop as monitors[0]
"""

import logging
from typing import Any

from ..config.schema import (
    Action,
    ClickActionConfig,
    DragActionConfig,
    HighlightActionConfig,
    MouseDownActionConfig,
    MouseMoveActionConfig,
    MouseUpActionConfig,
    ScrollActionConfig,
    TargetConfig,
)
from ..exceptions import ActionExecutionError
from ..hal.interfaces.input_controller import MouseButton as HALMouseButton
from .base import ActionExecutorBase
from .registry import register_executor

logger = logging.getLogger(__name__)


@register_executor
class MouseActionExecutor(ActionExecutorBase):
    """Executor for mouse-related actions.

    Handles all mouse operations including:
    - MOUSE_MOVE: Move cursor to position
    - MOUSE_DOWN: Press and hold button
    - MOUSE_UP: Release button
    - MOUSE_SCROLL / SCROLL: Scroll wheel
    - CLICK: Combined move + down + up with timing
    - DOUBLE_CLICK: Click with count=2
    - RIGHT_CLICK: Click with button=right
    - DRAG: Move + down + move + up

    All actions support both coordinate-based and image-based targeting.
    """

    def get_supported_action_types(self) -> list[str]:
        """Get list of mouse action types this executor handles.

        Returns:
            List of action type strings
        """
        return [
            "MOUSE_MOVE",
            "MOUSE_DOWN",
            "MOUSE_UP",
            "MOUSE_SCROLL",
            "SCROLL",
            "CLICK",
            "DOUBLE_CLICK",
            "RIGHT_CLICK",
            "DRAG",
            "HIGHLIGHT",
        ]

    def execute(self, action: Action, typed_config: Any) -> bool:
        """Execute mouse action with validated configuration.

        Args:
            action: Pydantic Action model
            typed_config: Type-specific validated configuration object

        Returns:
            bool: True if action succeeded

        Raises:
            ActionExecutionError: If action execution fails critically
        """
        # File-based debug logging at entry point - NO stderr output (breaks JSON protocol)
        import os
        import tempfile
        from datetime import datetime

        # Try multiple log paths
        debug_paths = [
            "/tmp/qontinui_mouse_executor_debug.log",
            os.path.join(tempfile.gettempdir(), "qontinui_mouse_executor_debug.log"),
            "/mnt/c/Users/jspin/Documents/qontinui_parent/mouse_executor_debug.log",
        ]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_msg = f"[{timestamp}] MouseActionExecutor.execute() called for {action.type} (ID: {action.id})\n"

        for debug_log_path in debug_paths:
            try:
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(log_msg)
                    f.write(
                        f"[{timestamp}]   Typed config type: {type(typed_config).__name__}\n"
                    )
                break  # Stop after first successful write
            except Exception:
                continue

        action_type = action.type
        logger.debug(f"Executing mouse action: {action_type}")

        try:
            if action_type == "MOUSE_MOVE":
                return self._execute_mouse_move(action, typed_config)
            elif action_type == "MOUSE_DOWN":
                return self._execute_mouse_down(action, typed_config)
            elif action_type == "MOUSE_UP":
                return self._execute_mouse_up(action, typed_config)
            elif action_type in ("MOUSE_SCROLL", "SCROLL"):
                return self._execute_scroll(action, typed_config)
            elif action_type == "CLICK":
                return self._execute_click(action, typed_config)
            elif action_type == "DOUBLE_CLICK":
                return self._execute_double_click(action, typed_config)
            elif action_type == "RIGHT_CLICK":
                return self._execute_right_click(action, typed_config)
            elif action_type == "DRAG":
                return self._execute_drag(action, typed_config)
            elif action_type == "HIGHLIGHT":
                return self._execute_highlight(action, typed_config)
            else:
                raise ActionExecutionError(
                    action_type=action_type,
                    reason=f"Unsupported action type: {action_type}",
                )

        except ActionExecutionError:
            raise
        except Exception as e:
            logger.error(f"Error executing {action_type}: {e}")
            raise ActionExecutionError(
                action_type=action_type, reason=f"Failed to execute {action_type}: {e}"
            ) from e

    # Helper methods

    def _get_target_location(
        self, target: TargetConfig | None
    ) -> tuple[int, int] | None:
        """Get target location from TargetConfig.

        Handles different target types:
        - ImageTarget: Find image on screen
        - CoordinatesTarget: Use specified coordinates
        - RegionTarget: Use center of region
        - CurrentPositionTarget: Return None (use current position)
        - LastFindResultTarget: Use first match from last action result
        - ResultIndexTarget: Use specific match from last action result by index
        - ResultByImageTarget: Use match from specific image in last action result
        - None: Return None (no target specified)

        Args:
            target: Target configuration or None

        Returns:
            Tuple of (x, y) coordinates or None if not found or current position
        """
        # File-based debug logging at entry
        import os
        import tempfile
        from datetime import datetime

        from ..config.models.targets import (
            AllResultsTarget,
            CoordinatesTarget,
            ImageTarget,
            LastFindResultTarget,
            RegionTarget,
            ResultByImageTarget,
            ResultIndexTarget,
        )

        debug_log_path = os.path.join(
            tempfile.gettempdir(), "qontinui_get_target_location_debug.log"
        )

        def log_debug(msg: str):
            """Write to debug file."""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            log_line = f"[{timestamp}] {msg}\n"
            try:
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(log_line)
            except Exception:
                pass

        log_debug("_get_target_location() called")
        log_debug(f"  target: {target}")
        log_debug(f"  target type: {type(target).__name__ if target else 'None'}")
        log_debug(f"  target is None: {target is None}")

        if target:
            log_debug(f"  hasattr(target, 'type'): {hasattr(target, 'type')}")
            if hasattr(target, "type"):
                log_debug(f"  target.type: {target.type}")
            log_debug(
                f"  isinstance(target, LastFindResultTarget): {isinstance(target, LastFindResultTarget)}"
            )
            log_debug(
                f"  isinstance(target, ImageTarget): {isinstance(target, ImageTarget)}"
            )
            log_debug(
                f"  isinstance(target, CoordinatesTarget): {isinstance(target, CoordinatesTarget)}"
            )
            log_debug(
                f"  isinstance(target, RegionTarget): {isinstance(target, RegionTarget)}"
            )

        logger.debug(f"_get_target_location() called with target={target}")
        logger.debug(f"Target type: {type(target).__name__ if target else 'None'}")

        # When target is None, no target specified
        if target is None:
            logger.debug("Target is None - no target specified")
            return None

        # Handle typed targets
        if isinstance(target, ImageTarget):
            # Use find wrapper to locate image
            # Note: ImageTarget now supports multiple image_ids, but for mouse actions
            # we use the first image_id for backward compatibility
            image_ids = target.image_ids
            if not image_ids:
                logger.error("ImageTarget has no image_ids")
                return None

            image_id = image_ids[0]  # Use first image for single target
            similarity = 0.8  # Default similarity

            if target.search_options and target.search_options.similarity is not None:
                similarity = target.search_options.similarity

            logger.debug(f"Finding image {image_id} with similarity {similarity}")

            image = self.context.config.image_map.get(image_id)
            if image and image.file_path:
                # Use find wrapper to locate image
                from ..wrappers.find_wrapper import Find  # type: ignore[attr-defined]

                location = Find.image(image.file_path, similarity=similarity)
                if location:
                    logger.debug(f"Found image at {location}")
                    return location  # type: ignore[no-any-return]
                else:
                    logger.warning(f"Image {image_id} not found on screen")
                    return None
            else:
                if not image:
                    logger.error(f"Image ID not found in image_map: {image_id}")
                else:
                    logger.error(f"Image file_path is None for image: {image_id}")
                return None

        elif isinstance(target, CoordinatesTarget):
            coords = target.coordinates
            logger.debug(f"Using coordinates: ({coords.x}, {coords.y})")
            return (coords.x, coords.y)

        elif isinstance(target, RegionTarget):
            region = target.region
            # Return center of region
            x = region.x + region.width // 2
            y = region.y + region.height // 2
            logger.debug(f"Using region center: ({x}, {y})")
            return (x, y)

        elif isinstance(target, LastFindResultTarget):
            # File-based debug logging for LastFindResultTarget
            import os
            import tempfile
            from datetime import datetime

            debug_log_path = os.path.join(
                tempfile.gettempdir(), "qontinui_last_find_result_debug.log"
            )

            def log_debug(msg: str):
                """Write to debug file."""
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                log_line = f"[{timestamp}] {msg}\n"
                logger.info(f"[LAST_FIND_RESULT_DEBUG] {msg}")
                try:
                    with open(debug_log_path, "a", encoding="utf-8") as f:
                        f.write(log_line)
                except Exception:
                    pass

            log_debug("Target is LastFindResultTarget")
            log_debug(f"self.context type: {type(self.context)}")
            log_debug(
                f"self.context has 'last_action_result': {hasattr(self.context, 'last_action_result')}"
            )

            if hasattr(self.context, "last_action_result"):
                log_debug(
                    f"self.context.last_action_result: {self.context.last_action_result}"
                )
                log_debug(
                    f"self.context.last_action_result type: {type(self.context.last_action_result)}"
                )
                log_debug(
                    f"self.context.last_action_result is None: {self.context.last_action_result is None}"
                )

                if self.context.last_action_result:
                    log_debug(
                        f"self.context.last_action_result.matches: {self.context.last_action_result.matches}"
                    )
                    log_debug(
                        f"matches count: {len(self.context.last_action_result.matches) if self.context.last_action_result.matches else 0}"
                    )
            else:
                log_debug("ERROR: context does not have 'last_action_result' attribute")

            logger.debug("Target is LastFindResultTarget")
            # Use first match from last action result
            if (
                self.context.last_action_result
                and self.context.last_action_result.matches
            ):
                match = self.context.last_action_result.matches[0]
                # =======================================================================
                # MONITOR OFFSET APPLICATION - Critical for Multi-Monitor Accuracy
                # =======================================================================
                #
                # The FIND action captures the ENTIRE virtual desktop (all monitors),
                # so match coordinates are relative to the virtual desktop origin.
                #
                # The offset values come from Rust (mcp_api.rs) and represent the
                # virtual desktop origin: (min_x, min_y) across all monitors.
                #
                # By adding the offset, we convert from screenshot-relative coordinates
                # to absolute virtual desktop coordinates that pyautogui can use.
                #
                # Example:
                #   match = (65, 1372)  # relative to screenshot
                #   offset = (-1920, 0)  # virtual desktop origin
                #   result = (-1855, 1372)  # absolute coordinates for pyautogui
                #
                # See module docstring and mcp_api.rs for full documentation.
                # =======================================================================
                offset_x = getattr(self.context, "monitor_offset_x", 0)
                offset_y = getattr(self.context, "monitor_offset_y", 0)
                location = (match.x + offset_x, match.y + offset_y)
                log_debug(
                    f"Using last action result match[0]: raw=({match.x}, {match.y}), "
                    f"offset=({offset_x}, {offset_y}), final={location}"
                )
                logger.debug(
                    f"Using last action result match[0]: raw=({match.x}, {match.y}), "
                    f"offset=({offset_x}, {offset_y}), final={location}"
                )
                return location
            else:
                log_debug(
                    "ERROR: LastFindResultTarget requested but no matches in last_action_result"
                )
                logger.error(
                    "LastFindResultTarget requested but no matches in last_action_result"
                )
                return None

        elif isinstance(target, ResultIndexTarget):
            logger.debug(f"Target is ResultIndexTarget with index={target.index}")
            # Access specific match by index
            if not self.context.last_action_result:
                logger.error(
                    "ResultIndexTarget requested but no last_action_result available"
                )
                return None

            matches = self.context.last_action_result.matches
            if not matches:
                logger.error(
                    "ResultIndexTarget requested but no matches in last_action_result"
                )
                return None

            if target.index < 0 or target.index >= len(matches):
                logger.error(
                    f"ResultIndexTarget index {target.index} out of bounds "
                    f"(available: 0-{len(matches)-1})"
                )
                return None

            match = matches[target.index]
            # Apply monitor offset to convert relative coordinates to absolute screen coordinates
            offset_x = getattr(self.context, "monitor_offset_x", 0)
            offset_y = getattr(self.context, "monitor_offset_y", 0)
            location = (match.x + offset_x, match.y + offset_y)
            logger.debug(
                f"Using match[{target.index}]: raw=({match.x}, {match.y}), "
                f"offset=({offset_x}, {offset_y}), final={location}"
            )
            return location

        elif isinstance(target, AllResultsTarget):
            logger.debug("Target is AllResultsTarget")
            # AllResultsTarget is not supported for single-location actions
            # Return first match location as fallback
            if (
                self.context.last_action_result
                and self.context.last_action_result.matches
            ):
                match = self.context.last_action_result.matches[0]
                # Apply monitor offset to convert relative coordinates to absolute screen coordinates
                offset_x = getattr(self.context, "monitor_offset_x", 0)
                offset_y = getattr(self.context, "monitor_offset_y", 0)
                location = (match.x + offset_x, match.y + offset_y)
                logger.warning(
                    "AllResultsTarget not supported for single-location actions, "
                    f"using first match: raw=({match.x}, {match.y}), "
                    f"offset=({offset_x}, {offset_y}), final={location}"
                )
                return location
            else:
                logger.error(
                    "AllResultsTarget requested but no matches in last_action_result"
                )
                return None

        elif isinstance(target, ResultByImageTarget):
            logger.debug(
                f"Target is ResultByImageTarget with image_id={target.image_id}"
            )
            # Find match from specific source image
            if not self.context.last_action_result:
                logger.error(
                    "ResultByImageTarget requested but no last_action_result available"
                )
                return None

            matches = self.context.last_action_result.matches
            if not matches:
                logger.error(
                    "ResultByImageTarget requested but no matches in last_action_result"
                )
                return None

            # Search for match with matching source_image_id in metadata
            for match in matches:
                # Check if match has metadata with source_image_id
                if hasattr(match, "match_object") and hasattr(
                    match.match_object, "metadata"
                ):
                    metadata = match.match_object.metadata
                    # Check both as attribute and dict key for flexibility
                    source_id = getattr(metadata, "source_image_id", None)
                    if source_id is None and hasattr(metadata, "__dict__"):
                        source_id = metadata.__dict__.get("source_image_id")

                    if source_id == target.image_id:
                        # Apply monitor offset to convert relative coordinates to absolute
                        offset_x = getattr(self.context, "monitor_offset_x", 0)
                        offset_y = getattr(self.context, "monitor_offset_y", 0)
                        location = (match.x + offset_x, match.y + offset_y)
                        logger.debug(
                            f"Found match from source image '{target.image_id}': "
                            f"raw=({match.x}, {match.y}), offset=({offset_x}, {offset_y}), "
                            f"final={location}"
                        )
                        return location

            logger.error(
                f"ResultByImageTarget: no match found with source_image_id='{target.image_id}'"
            )
            return None

        elif hasattr(target, "type") and target.type == "currentPosition":
            # CurrentPositionTarget - use current mouse position
            logger.debug("Using current position (pure action)")
            return None

        logger.error(f"Unsupported target type: {type(target)}")
        return None

    def _convert_mouse_button(self, button_str: str | None) -> HALMouseButton:
        """Convert button string to HAL MouseButton enum.

        Args:
            button_str: Button name ("LEFT", "RIGHT", "MIDDLE") or None

        Returns:
            HAL MouseButton enum value
        """
        if button_str is None:
            return HALMouseButton.LEFT

        button_map = {
            "LEFT": HALMouseButton.LEFT,
            "RIGHT": HALMouseButton.RIGHT,
            "MIDDLE": HALMouseButton.MIDDLE,
        }
        return button_map.get(button_str.upper(), HALMouseButton.LEFT)

    # Pure action executors

    def _execute_mouse_move(
        self, action: Action, typed_config: MouseMoveActionConfig
    ) -> bool:
        """Execute MOUSE_MOVE action (pure) - move mouse to position.

        Args:
            action: Action model
            typed_config: Validated MouseMoveActionConfig

        Returns:
            True if successful
        """
        # File-based debug logging
        import os
        import tempfile
        from datetime import datetime

        debug_log_path = os.path.join(
            tempfile.gettempdir(), "qontinui_mouse_move_debug.log"
        )

        def log_debug(msg: str):
            """Write to both logger and debug file."""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            log_line = f"[{timestamp}] {msg}\n"
            logger.info(f"[MOUSE_MOVE_DEBUG] {msg}")
            try:
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(log_line)
            except Exception:
                pass

        log_debug(f"_execute_mouse_move() called for action {action.id}")
        log_debug(f"Target config: {typed_config.target}")
        log_debug(f"Target type: {type(typed_config.target).__name__}")

        try:
            log_debug("About to call _get_target_location()")
            location = self._get_target_location(typed_config.target)
            log_debug(f"_get_target_location() returned: {location}")
        except Exception as e:
            log_debug(f"EXCEPTION in _get_target_location(): {type(e).__name__}: {e}")
            import traceback

            log_debug(f"Traceback: {traceback.format_exc()}")
            raise

        if not location:
            log_debug("ERROR: Failed to get target location for MOUSE_MOVE")
            logger.error("Failed to get target location for MOUSE_MOVE")
            return False

        # Get duration from config (convert milliseconds to seconds)
        duration_ms = typed_config.move_duration or 0
        duration_seconds = duration_ms / 1000.0 if duration_ms else 0.0

        # If moveInstantly is True, override duration
        if typed_config.move_instantly:
            duration_seconds = 0.0
            log_debug("moveInstantly=True, duration set to 0")

        log_debug(
            f"Moving mouse to {location} (duration: {duration_ms}ms = {duration_seconds}s)"
        )
        logger.info(f"Moving mouse to {location} (duration: {duration_ms}ms)")
        success = self.context.mouse.move(location[0], location[1], duration_seconds)
        log_debug(f"context.mouse.move() returned: {success}")

        if success:
            log_debug(f"SUCCESS: Mouse moved to {location}")
            logger.debug(f"[PURE ACTION] Mouse moved to {location}")
        else:
            log_debug("FAILED: Mouse move failed")
            logger.error(f"Failed to move mouse to {location}")

        log_debug(f"Debug log written to: {debug_log_path}")
        return success  # type: ignore[no-any-return]

    def _execute_mouse_down(
        self, action: Action, typed_config: MouseDownActionConfig
    ) -> bool:
        """Execute MOUSE_DOWN action (pure) - press and hold mouse button.

        Args:
            action: Action model
            typed_config: Validated MouseDownActionConfig

        Returns:
            True if successful
        """
        # Convert button type
        button = self._convert_mouse_button(
            typed_config.mouse_button.value if typed_config.mouse_button else None
        )

        # Get optional position
        location = None
        if typed_config.target:
            location = self._get_target_location(typed_config.target)
        elif typed_config.coordinates:
            location = (typed_config.coordinates.x, typed_config.coordinates.y)

        if location:
            logger.info(f"Pressing {button.value} button at {location}")
            success = self.context.mouse.down(location[0], location[1], button)
        else:
            logger.info(f"Pressing {button.value} button at current position")
            success = self.context.mouse.down(button=button)

        if success:
            logger.debug(f"[PURE ACTION] Mouse {button.value} button pressed")
        else:
            logger.error(f"Failed to press {button.value} button")

        return success  # type: ignore[no-any-return]

    def _execute_mouse_up(
        self, action: Action, typed_config: MouseUpActionConfig
    ) -> bool:
        """Execute MOUSE_UP action (pure) - release mouse button.

        Args:
            action: Action model
            typed_config: Validated MouseUpActionConfig

        Returns:
            True if successful
        """
        # Convert button type
        button = self._convert_mouse_button(
            typed_config.mouse_button.value if typed_config.mouse_button else None
        )

        # Get optional position
        location = None
        if typed_config.target:
            location = self._get_target_location(typed_config.target)
        elif typed_config.coordinates:
            location = (typed_config.coordinates.x, typed_config.coordinates.y)

        if location:
            logger.info(f"Releasing {button.value} button at {location}")
            success = self.context.mouse.up(location[0], location[1], button)
        else:
            logger.info(f"Releasing {button.value} button at current position")
            success = self.context.mouse.up(button=button)

        if success:
            logger.debug(f"[PURE ACTION] Mouse {button.value} button released")
        else:
            logger.error(f"Failed to release {button.value} button")

        return success  # type: ignore[no-any-return]

    def _execute_scroll(self, action: Action, typed_config: ScrollActionConfig) -> bool:
        """Execute SCROLL/MOUSE_SCROLL action - scroll mouse wheel.

        Args:
            action: Action model
            typed_config: Validated ScrollActionConfig

        Returns:
            True if successful
        """
        # Get scroll direction and distance
        direction = typed_config.direction
        clicks = typed_config.clicks or 3  # Default to 3 clicks

        # Convert direction to scroll amount (positive for up, negative for down)
        scroll_amount = clicks if direction == "up" else -clicks

        # Get optional target location
        location = None
        if typed_config.target:
            location = self._get_target_location(typed_config.target)
            if location:
                # Move to location before scrolling
                self.context.mouse.move(location[0], location[1], 0.0)

        logger.info(f"Scrolling {direction} by {clicks} clicks")
        success = self.context.mouse.scroll(scroll_amount)

        if success:
            logger.debug(f"[PURE ACTION] Scrolled {direction} by {clicks}")
        else:
            logger.error(f"Failed to scroll {direction}")

        return success  # type: ignore[no-any-return]

    # Combined action executors

    def _execute_click(self, action: Action, typed_config: ClickActionConfig) -> bool:
        """Execute CLICK action - combined move + down + up with timing.

        This is a COMBINED action that orchestrates pure actions.

        Args:
            action: Action model
            typed_config: Validated ClickActionConfig

        Returns:
            True if successful
        """
        # DEBUG: Write to file to confirm this method is called
        import os
        import tempfile
        from datetime import datetime

        debug_log_path = os.path.join(
            tempfile.gettempdir(), "qontinui_click_executor_debug.log"
        )

        def debug_write(msg: str) -> None:
            try:
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"[{timestamp}] {msg}\n")
            except Exception:
                pass

        debug_write(f"_execute_click() called for action {action.id}")
        debug_write(f"Target: {typed_config.target}")
        debug_write(f"Mouse button: {typed_config.mouse_button}")

        logger.info("[COMBINED ACTION: CLICK] Starting click action")
        debug_write("Starting click action")

        # Check if target is None or currentPosition (pure action)
        location = None
        if typed_config.target and typed_config.target.type != "currentPosition":
            debug_write(f"Getting target location for: {typed_config.target}")
            location = self._get_target_location(typed_config.target)
            debug_write(f"Got location: {location}")
            if not location:
                logger.error("Failed to get target location for CLICK")
                debug_write("ERROR: Failed to get target location - returning False")
                return False
        else:
            logger.debug(
                "No target specified - clicking at current position (pure action)"
            )
            debug_write("No target - clicking at current position")

        # Get button
        button_value = (
            typed_config.mouse_button.value if typed_config.mouse_button else None
        )
        logger.info(
            f"[CLICK DEBUG] Raw mouse_button from config: {typed_config.mouse_button}, value: {button_value}"
        )
        button = self._convert_mouse_button(button_value)
        logger.info(f"[CLICK DEBUG] Converted button: {button}, type: {type(button)}")
        debug_write(f"Button converted: {button}")

        # Get click count
        click_count = typed_config.number_of_clicks or 1
        debug_write(f"Click count: {click_count}")

        # Get timing values (convert milliseconds to seconds)
        hold_duration = (
            typed_config.press_duration
            or self._get_default_timing("mouse", "click_hold_duration", 100)
        ) / 1000.0
        release_delay = (
            typed_config.pause_after_release
            or self._get_default_timing("mouse", "click_release_delay", 0)
        ) / 1000.0

        # Safety settings
        safety_release = self._get_default_timing("mouse", "click_safety_release", True)
        safety_delay = self._get_default_timing("mouse", "safety_release_delay", 0.1)

        # Get current position for logging
        debug_write("Getting current mouse position...")
        current_pos = self.context.mouse.position()
        debug_write(f"Current position: ({current_pos.x}, {current_pos.y})")
        logger.debug(f"Current position: ({current_pos.x}, {current_pos.y})")
        logger.debug(
            f"Target: {location}, Button: {button.value}, Count: {click_count}"
        )
        logger.debug(f"Timing: hold={hold_duration}s, release={release_delay}s")

        # Step 1: Safety - Release all buttons first (if enabled)
        if safety_release:
            logger.debug("Step 1: Release all mouse buttons")
            debug_write("Step 1: Safety release - calling mouse.up() 3 times")
            self.context.mouse.up(button=HALMouseButton.LEFT)
            debug_write("  - Released LEFT")
            self.context.mouse.up(button=HALMouseButton.RIGHT)
            debug_write("  - Released RIGHT")
            self.context.mouse.up(button=HALMouseButton.MIDDLE)
            debug_write("  - Released MIDDLE")
            self.context.time.wait(safety_delay)
            debug_write("  - Safety delay completed")

        # Step 1.5: Move to target location if specified
        if location:
            logger.debug(f"Step 1.5: Moving to target location: {location}")
            debug_write(f"Step 1.5: Moving mouse to ({location[0]}, {location[1]})")
            self.context.mouse.move(location[0], location[1])
            debug_write("  - Move command issued")
            # Small delay after move to ensure GUI registers position
            self.context.time.wait(0.05)
            debug_write("  - Post-move delay completed")

        # Step 2: Perform clicks at target/current position
        click_location = location if location else "current position"
        logger.info(
            f"[CLICK DEBUG] Step 2: Perform {click_count} click(s) at {click_location} with button {button}"
        )
        debug_write(f"Step 2: Starting {click_count} click(s) at {click_location}")
        for i in range(click_count):
            logger.info(f"[CLICK DEBUG] Click {i+1}/{click_count} - Pressing {button}")
            debug_write(f"  Click {i+1}/{click_count}")

            # Sub-action: MOUSE_DOWN
            logger.info(f"[CLICK DEBUG] Calling mouse.down with button={button}")
            debug_write(f"    - Calling mouse.down(button={button})")
            self.context.mouse.down(button=button)
            debug_write("    - mouse.down returned")
            self.context.time.wait(hold_duration)
            debug_write("    - hold delay completed")

            # Sub-action: MOUSE_UP
            logger.info(f"[CLICK DEBUG] Calling mouse.up with button={button}")
            debug_write(f"    - Calling mouse.up(button={button})")
            self.context.mouse.up(button=button)
            debug_write("    - mouse.up returned")

            if i < click_count - 1:
                self.context.time.wait(release_delay)
                debug_write("    - inter-click delay completed")

        # Step 3: Final safety release
        self.context.time.wait(release_delay)
        self.context.mouse.up(button=button)
        logger.debug("Step 3: Final safety release")
        debug_write("Step 3: Final safety release completed")

        logger.info(
            f"[COMBINED ACTION: CLICK] Completed {click_count} {button.value}-click(s)"
        )
        debug_write(f"SUCCESS: Completed {click_count} {button.value}-click(s)")
        return True

    def _execute_double_click(
        self, action: Action, typed_config: ClickActionConfig
    ) -> bool:
        """Execute DOUBLE_CLICK action - convenience wrapper for CLICK with count=2.

        Args:
            action: Action model
            typed_config: Validated ClickActionConfig

        Returns:
            True if successful
        """
        logger.info("[COMBINED ACTION: DOUBLE_CLICK] Delegating to CLICK with count=2")

        # Create modified config with count=2
        from ..config.schema import ClickActionConfig

        double_click_config = ClickActionConfig(
            target=typed_config.target if typed_config else None,
            number_of_clicks=2,
            mouse_button=typed_config.mouse_button if typed_config else None,
            press_duration=typed_config.press_duration if typed_config else None,
            pause_after_press=typed_config.pause_after_press if typed_config else None,
            pause_after_release=(
                typed_config.pause_after_release if typed_config else None
            ),
        )

        return self._execute_click(action, double_click_config)

    def _execute_right_click(
        self, action: Action, typed_config: ClickActionConfig
    ) -> bool:
        """Execute RIGHT_CLICK action - convenience wrapper for CLICK with button=right.

        Args:
            action: Action model
            typed_config: Validated ClickActionConfig

        Returns:
            True if successful
        """
        logger.info(
            "[COMBINED ACTION: RIGHT_CLICK] Delegating to CLICK with button=right"
        )

        # Create modified config with button=right
        from ..config.schema import ClickActionConfig, MouseButton

        right_click_config = ClickActionConfig(
            target=typed_config.target if typed_config else None,
            number_of_clicks=typed_config.number_of_clicks if typed_config else 1,
            mouse_button=MouseButton.RIGHT,
            press_duration=typed_config.press_duration if typed_config else None,
            pause_after_press=typed_config.pause_after_press if typed_config else None,
            pause_after_release=(
                typed_config.pause_after_release if typed_config else None
            ),
        )

        return self._execute_click(action, right_click_config)

    def _execute_drag(self, action: Action, typed_config: DragActionConfig) -> bool:
        """Execute DRAG action - move + down + move + up.

        This is a COMBINED action using pure actions:
        1. Move to start position
        2. Press button
        3. Move to end position (with button held)
        4. Release button

        Args:
            action: Action model
            typed_config: Validated DragActionConfig

        Returns:
            True if successful
        """
        logger.info("[COMBINED ACTION: DRAG] Starting drag action")

        # Get start location
        start = self._get_target_location(typed_config.source)
        if not start:
            logger.error("Failed to get start location for DRAG")
            return False

        # Get destination - can be TargetConfig, Coordinates, or Region
        end = None
        if isinstance(typed_config.destination, dict):
            # It's a Coordinates or Region dict
            end_x = typed_config.destination.get("x", 0)
            end_y = typed_config.destination.get("y", 0)
            end = (end_x, end_y)
        else:
            # It's a TargetConfig - get location
            end = self._get_target_location(typed_config.destination)  # type: ignore[arg-type]

        if not end:
            logger.error("Failed to get destination location for DRAG")
            return False

        # Get timing values (convert milliseconds to seconds)
        duration = (
            typed_config.drag_duration
            or self._get_default_timing("mouse", "drag_default_duration", 500)
        ) / 1000.0
        start_delay = (
            typed_config.delay_before_move
            or self._get_default_timing("mouse", "drag_start_delay", 100)
        ) / 1000.0
        end_delay = (
            typed_config.delay_after_drag
            or self._get_default_timing("mouse", "drag_end_delay", 100)
        ) / 1000.0

        logger.debug(f"From {start} to {end}")
        logger.debug(
            f"Timing: duration={duration}s, start_delay={start_delay}s, end_delay={end_delay}s"
        )

        # Step 1: Move to start position
        logger.debug("Step 1: Move to start")
        self.context.mouse.move(start[0], start[1], 0.0)

        # Step 2: Press button
        self.context.time.wait(start_delay)
        logger.debug("Step 2: Press left button")
        self.context.mouse.down(button=HALMouseButton.LEFT)

        # Step 3: Move to end position (dragging)
        self.context.time.wait(start_delay)
        logger.debug("Step 3: Move to end (dragging)")
        self.context.mouse.move(end[0], end[1], duration)

        # Step 4: Release button
        self.context.time.wait(end_delay)
        logger.debug("Step 4: Release left button")
        self.context.mouse.up(button=HALMouseButton.LEFT)

        logger.info("[COMBINED ACTION: DRAG] Completed")
        return True

    def _execute_highlight(
        self, action: Action, typed_config: HighlightActionConfig
    ) -> bool:
        """Execute HIGHLIGHT action - visually highlight a screen region.

        Args:
            action: Action model
            typed_config: Validated HighlightActionConfig

        Returns:
            True if successful
        """
        from .highlight_overlay import HighlightOverlay

        # Get target location
        location = self._get_target_location(typed_config.target)
        if not location:
            logger.error("Failed to get target location for HIGHLIGHT")
            return False

        # Extract configuration with defaults
        duration = typed_config.duration or 2000  # Default 2 seconds
        color = typed_config.color or "#FF0000"  # Default red
        thickness = typed_config.thickness or 3  # Default 3px
        style = typed_config.style or "box"  # Default box style

        logger.info(
            f"Highlighting at ({location[0]}, {location[1]}) - "
            f"duration={duration}ms, color={color}, thickness={thickness}px, style={style}"
        )

        try:
            # Create and show the highlight overlay
            overlay = HighlightOverlay(
                x=location[0],
                y=location[1],
                duration_ms=duration,
                color=color,
                thickness=thickness,
                style=style,
            )
            overlay.show()

            logger.debug(
                f"[HIGHLIGHT] Displaying {style} at {location} "
                f"with color {color}, thickness {thickness}px for {duration}ms"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to create highlight overlay: {e}")
            return False
