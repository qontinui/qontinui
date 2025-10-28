"""Mouse action executor for all mouse-related actions.

This module provides the MouseActionExecutor class that handles all mouse operations
including movement, clicking, dragging, and scrolling.
"""

import logging
from typing import Any

from ..config.schema import (
    Action,
    ClickActionConfig,
    DragActionConfig,
    MouseDownActionConfig,
    MouseMoveActionConfig,
    MouseUpActionConfig,
    ScrollActionConfig,
    TargetConfig,
)
from ..exceptions import ActionExecutionError
from ..hal.interfaces.input_controller import MouseButton as HALMouseButton
from .base import ActionExecutorBase, ExecutionContext
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
            else:
                raise ActionExecutionError(f"Unsupported action type: {action_type}")

        except ActionExecutionError:
            raise
        except Exception as e:
            logger.error(f"Error executing {action_type}: {e}")
            raise ActionExecutionError(f"Failed to execute {action_type}: {e}") from e

    # Helper methods

    def _get_target_location(self, target: TargetConfig | None) -> tuple[int, int] | None:
        """Get target location from TargetConfig.

        Handles different target types:
        - ImageTarget: Find image on screen
        - CoordinatesTarget: Use specified coordinates
        - RegionTarget: Use center of region
        - CurrentPositionTarget: Return None (use current position)
        - "Last Find Result": Use stored last find location

        Args:
            target: Target configuration or None

        Returns:
            Tuple of (x, y) coordinates or None if not found or current position
        """
        from ..config.schema import CoordinatesTarget, ImageTarget, RegionTarget

        if target is None:
            return None

        # Handle string target for backward compatibility
        if isinstance(target, str) and target == "Last Find Result":
            if self.context.last_find_location:
                logger.debug(f"Using Last Find Result: {self.context.last_find_location}")
                return self.context.last_find_location
            else:
                logger.error("Last Find Result requested but no previous find result available")
                return None

        # Handle typed targets
        if isinstance(target, ImageTarget):
            # Use find wrapper to locate image
            image_id = target.image_id
            similarity = 0.8  # Default similarity

            if target.search_options and target.search_options.similarity is not None:
                similarity = target.search_options.similarity

            logger.debug(f"Finding image {image_id} with similarity {similarity}")

            if image_id:
                image = self.context.config.image_map.get(image_id)
                if image and image.file_path:
                    # Use find wrapper to locate image
                    from ..wrappers.find_wrapper import Find

                    location = Find.image(image.file_path, similarity=similarity)
                    if location:
                        logger.debug(f"Found image at {location}")
                        # Store as last find location
                        self.context.update_last_find_location(location)
                        return location
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
        location = self._get_target_location(typed_config.target)
        if not location:
            logger.error("Failed to get target location for MOUSE_MOVE")
            return False

        # Get duration from config (convert milliseconds to seconds)
        duration_ms = typed_config.move_duration or 0
        duration_seconds = duration_ms / 1000.0 if duration_ms else 0.0

        # If moveInstantly is True, override duration
        if typed_config.move_instantly:
            duration_seconds = 0.0

        logger.info(f"Moving mouse to {location} (duration: {duration_ms}ms)")
        success = self.context.mouse.move(location[0], location[1], duration_seconds)

        if success:
            logger.debug(f"[PURE ACTION] Mouse moved to {location}")
        else:
            logger.error(f"Failed to move mouse to {location}")

        return success

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

        return success

    def _execute_mouse_up(self, action: Action, typed_config: MouseUpActionConfig) -> bool:
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

        return success

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

        return success

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
        logger.info("[COMBINED ACTION: CLICK] Starting click action")

        # Check if target is None or currentPosition (pure action)
        location = None
        if typed_config.target and typed_config.target.type != "currentPosition":
            location = self._get_target_location(typed_config.target)
            if not location:
                logger.error("Failed to get target location for CLICK")
                return False
        else:
            logger.debug("No target specified - clicking at current position (pure action)")

        # Get button
        button = self._convert_mouse_button(
            typed_config.mouse_button.value if typed_config.mouse_button else None
        )

        # Get click count
        click_count = typed_config.number_of_clicks or 1

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
        current_pos = self.context.mouse.position()
        logger.debug(f"Current position: ({current_pos.x}, {current_pos.y})")
        logger.debug(
            f"Target: {location}, Button: {button.value}, Count: {click_count}"
        )
        logger.debug(f"Timing: hold={hold_duration}s, release={release_delay}s")

        # Step 1: Safety - Release all buttons first (if enabled)
        if safety_release:
            logger.debug("Step 1: Release all mouse buttons")
            self.context.mouse.up(button=HALMouseButton.LEFT)
            self.context.mouse.up(button=HALMouseButton.RIGHT)
            self.context.mouse.up(button=HALMouseButton.MIDDLE)
            self.context.time.wait(safety_delay)

        # Step 2: Perform clicks at current position
        logger.debug(f"Step 2: Perform {click_count} click(s) at current position")
        for i in range(click_count):
            logger.debug(f"Click {i+1}/{click_count}")

            # Sub-action: MOUSE_DOWN
            self.context.mouse.down(button=button)
            self.context.time.wait(hold_duration)

            # Sub-action: MOUSE_UP
            self.context.mouse.up(button=button)

            if i < click_count - 1:
                self.context.time.wait(release_delay)

        # Step 3: Final safety release
        self.context.time.wait(release_delay)
        self.context.mouse.up(button=button)
        logger.debug("Step 3: Final safety release")

        logger.info(f"[COMBINED ACTION: CLICK] Completed {click_count} {button.value}-click(s)")
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
            pause_after_release=typed_config.pause_after_release if typed_config else None,
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
        logger.info("[COMBINED ACTION: RIGHT_CLICK] Delegating to CLICK with button=right")

        # Create modified config with button=right
        from ..config.schema import ClickActionConfig, MouseButton

        right_click_config = ClickActionConfig(
            target=typed_config.target if typed_config else None,
            number_of_clicks=typed_config.number_of_clicks if typed_config else 1,
            mouse_button=MouseButton.RIGHT,
            press_duration=typed_config.press_duration if typed_config else None,
            pause_after_press=typed_config.pause_after_press if typed_config else None,
            pause_after_release=typed_config.pause_after_release if typed_config else None,
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
            end = self._get_target_location(typed_config.destination)

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
