"""ActionExecution - manages action lifecycle and execution.

This class handles the complete lifecycle of action execution including
timing, logging, and cross-cutting concerns.
"""

import logging
import time
from datetime import timedelta
from typing import Any

from .action_config import ActionConfig
from .action_interface import ActionInterface
from .action_result import ActionResult
from .object_collection import ObjectCollection

logger = logging.getLogger(__name__)


class ActionExecution:
    """Manages the execution lifecycle of actions.

    Port of Brobot's ActionExecution class.

    This class wraps action execution with lifecycle management including:
    - Pre-action pauses
    - Post-action pauses
    - Timing measurements
    - Logging
    - Error handling
    - Success/failure tracking

    All actions should go through this class to ensure consistent behavior
    across the framework, matching Brobot's approach where every action
    follows the same lifecycle pattern.
    """

    def __init__(self) -> None:
        """Initialize ActionExecution."""
        self._execution_count = 0
        self._success_count = 0
        self._failure_count = 0
        logger.debug("ActionExecution initialized")

    def perform(
        self,
        action: ActionInterface,
        action_description: str,
        action_config: ActionConfig,
        object_collections: tuple[ObjectCollection, ...],
    ) -> ActionResult:
        """Execute an action with full lifecycle management.

        This method implements the complete action lifecycle:
        1. Initialize result
        2. Apply pre-action pause
        3. Execute the action
        4. Apply post-action pause
        5. Record timing and success
        6. Handle errors

        Args:
            action: The action implementation to execute
            action_description: Human-readable description for logging
            action_config: Configuration controlling timing and behavior
            object_collections: Target objects for the action

        Returns:
            ActionResult containing execution results and timing
        """
        # Initialize result
        result = ActionResult(action_config)  # type: ignore[arg-type]
        object.__setattr__(result, "action_description", action_description)
        start_time = time.time()

        try:
            # Log action start
            action_name = action.__class__.__name__
            if action_description:
                logger.info(f"Executing {action_name}: {action_description}")
            else:
                logger.debug(f"Executing {action_name}")

            # Pre-action pause
            pause_before = action_config.get_pause_before_begin()
            if pause_before > 0:
                logger.debug(f"Pausing {pause_before}s before action")
                time.sleep(pause_before)

            # Execute the action
            self._execution_count += 1
            action.perform(result, *object_collections)

            # Check success
            if result.success:
                self._success_count += 1
                logger.debug(f"{action_name} completed successfully")
            else:
                self._failure_count += 1
                logger.warning(f"{action_name} failed")

            # Post-action pause
            pause_after = action_config.get_pause_after_end()
            if pause_after > 0:
                logger.debug(f"Pausing {pause_after}s after action")
                time.sleep(pause_after)

        except Exception as e:
            self._failure_count += 1
            logger.error(f"Action execution failed: {e}", exc_info=True)
            object.__setattr__(result, "success", False)
            object.__setattr__(result, "output_text", f"Error: {str(e)}")

        finally:
            # Record timing
            end_time = time.time()
            duration = end_time - start_time
            object.__setattr__(result, "duration", timedelta(seconds=duration))

            # Apply custom success criteria if provided
            success_criteria = action_config.get_success_criteria()
            if success_criteria:
                try:
                    custom_success = success_criteria(result)  # type: ignore[arg-type]
                    object.__setattr__(result, "success", custom_success)
                except Exception as e:
                    logger.error(f"Success criteria evaluation failed: {e}")

        return result

    def get_metrics(self) -> dict[str, Any]:
        """Get execution metrics.

        Returns:
            Dictionary containing execution statistics
        """
        total = max(1, self._execution_count)
        return {
            "total_executions": self._execution_count,
            "successful": self._success_count,
            "failed": self._failure_count,
            "success_rate": (self._success_count / total) * 100,
        }

    def reset_metrics(self):
        """Reset execution metrics."""
        self._execution_count = 0
        self._success_count = 0
        self._failure_count = 0
        logger.info("ActionExecution metrics reset")
