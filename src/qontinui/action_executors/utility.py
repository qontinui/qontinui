"""Utility action executors for WAIT and SCREENSHOT actions.

This module provides specialized executors for utility actions that don't fit
into other categories like mouse, keyboard, or vision actions.
"""

import logging
from typing import Any

from ..config.schema import Action, ScreenshotActionConfig, WaitActionConfig
from ..exceptions import (
    ActionExecutionError,
    ScreenCaptureException,
)
from .base import ActionExecutorBase, ExecutionContext
from .registry import register_executor

logger = logging.getLogger(__name__)


@register_executor
class UtilityActionExecutor(ActionExecutorBase):
    """Executor for utility actions: WAIT and SCREENSHOT.

    This executor handles general-purpose utility actions that support
    automation workflows but don't directly interact with UI elements.

    Supported actions:
        - WAIT: Pause execution for a specified duration
        - SCREENSHOT: Capture screen or region to file

    Example:
        >>> context = ExecutionContext(...)
        >>> executor = UtilityActionExecutor(context)
        >>>
        >>> # Execute WAIT action
        >>> wait_action = Action(type="WAIT", config={"duration": 1000})
        >>> wait_config = WaitActionConfig(duration=1000)
        >>> executor.execute(wait_action, wait_config)
        True
        >>>
        >>> # Execute SCREENSHOT action
        >>> screenshot_action = Action(type="SCREENSHOT", config={...})
        >>> screenshot_config = ScreenshotActionConfig(...)
        >>> executor.execute(screenshot_action, screenshot_config)
        True
    """

    def get_supported_action_types(self) -> list[str]:
        """Get list of action types handled by this executor.

        Returns:
            List containing "WAIT" and "SCREENSHOT"
        """
        return ["WAIT", "SCREENSHOT"]

    def execute(self, action: Action, typed_config: Any) -> bool:
        """Execute utility action with validated configuration.

        Args:
            action: Pydantic Action model with type, id, config
            typed_config: Type-specific validated configuration object
                - WaitActionConfig for WAIT actions
                - ScreenshotActionConfig for SCREENSHOT actions

        Returns:
            bool: True if action succeeded, False otherwise

        Raises:
            ActionExecutionError: If action execution fails critically
        """
        logger.debug(f"Executing utility action: {action.type}")

        if action.type == "WAIT":
            return self._execute_wait(action, typed_config)
        elif action.type == "SCREENSHOT":
            return self._execute_screenshot(action, typed_config)
        else:
            logger.error(f"Unknown utility action type: {action.type}")
            raise ActionExecutionError(
                action_type=action.type,
                reason=f"Utility executor does not handle {action.type}",
                action_id=action.id
            )

    def _execute_wait(self, action: Action, typed_config: WaitActionConfig) -> bool:
        """Execute WAIT action - pause execution for specified duration.

        Args:
            action: Pydantic Action model
            typed_config: Pre-validated WaitActionConfig

        Returns:
            bool: True if wait completed successfully

        Raises:
            ActionExecutionError: If duration is invalid
        """
        logger.debug("Executing WAIT action")

        # Get duration from typed config
        if typed_config.wait_for == "time":
            duration_ms = typed_config.duration

            if duration_ms is None:
                raise ActionExecutionError(
                    action_type="WAIT",
                    reason="Duration is required for wait_for='time'",
                    action_id=action.id
                )

            if duration_ms < 0:
                raise ActionExecutionError(
                    action_type="WAIT",
                    reason=f"Duration must be non-negative, got {duration_ms}",
                    action_id=action.id
                )

            # Convert milliseconds to seconds
            duration_seconds = duration_ms / 1000.0

            logger.info(f"Waiting {duration_ms}ms ({duration_seconds}s)")
            self.context.time.wait(duration_seconds)

            self._emit_action_success(
                action,
                {"duration_ms": duration_ms, "duration_seconds": duration_seconds}
            )

            return True
        else:
            # Other wait types (target, state, condition) not yet implemented
            raise ActionExecutionError(
                action_type="WAIT",
                reason=f"wait_for='{typed_config.wait_for}' not yet implemented",
                action_id=action.id
            )

    def _execute_screenshot(self, action: Action, typed_config: ScreenshotActionConfig) -> bool:
        """Execute SCREENSHOT action - capture screen or region to file.

        Args:
            action: Pydantic Action model
            typed_config: Pre-validated ScreenshotActionConfig

        Returns:
            bool: True if screenshot was captured successfully

        Raises:
            ScreenCaptureException: If screen capture fails
            ActionExecutionError: If save operation fails
        """
        logger.debug("Executing SCREENSHOT action")

        try:
            # Generate default filename using timestamp
            timestamp = int(self.context.time.now().timestamp())
            default_filename = f"screenshot_{timestamp}.png"

            # Get filename from config
            filename = default_filename
            if typed_config.save_to_file and typed_config.save_to_file.enabled:
                if typed_config.save_to_file.filename:
                    filename = typed_config.save_to_file.filename

                # If directory specified, prepend it
                if typed_config.save_to_file.directory:
                    import os
                    filename = os.path.join(typed_config.save_to_file.directory, filename)

            # Check if region is specified
            if typed_config.region:
                region = typed_config.region
                logger.debug(
                    f"Capturing region: x={region.x}, y={region.y}, "
                    f"width={region.width}, height={region.height}"
                )

                # Capture specific region
                self.context.screen.save(
                    filename,
                    region=(region.x, region.y, region.width, region.height)
                )

                self._emit_action_success(
                    action,
                    {
                        "filename": filename,
                        "region": {
                            "x": region.x,
                            "y": region.y,
                            "width": region.width,
                            "height": region.height
                        }
                    }
                )
            else:
                logger.debug("Capturing full screen")

                # Capture full screen
                self.context.screen.save(filename)

                self._emit_action_success(
                    action,
                    {"filename": filename, "fullscreen": True}
                )

            logger.info(f"Screenshot saved to {filename}")
            return True

        except ScreenCaptureException:
            # Re-raise screen capture errors as-is
            raise
        except Exception as e:
            # Wrap other exceptions
            logger.error(f"Failed to save screenshot: {e}", exc_info=True)
            raise ActionExecutionError(
                action_type="SCREENSHOT",
                reason=f"Failed to save screenshot: {e}",
                action_id=action.id
            ) from e
