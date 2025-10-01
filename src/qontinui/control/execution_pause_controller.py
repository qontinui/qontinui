"""Execution pause controller - ported from Qontinui framework.

Manages pause points and pause behavior during execution.
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from .execution_controller import ExecutionController

logger = logging.getLogger(__name__)


class PausePoint:
    """Represents a pause point in the execution flow.

    Pause points are strategic locations where execution can be
    paused for debugging or user intervention.
    """

    def __init__(self, name: str, description: str = "", enabled: bool = True):
        """Initialize a pause point.

        Args:
            name: Unique name for the pause point
            description: Human-readable description
            enabled: Whether this pause point is active
        """
        self.name = name
        self.description = description
        self.enabled = enabled
        self.hit_count = 0
        self.last_hit_time: float | None = None

    def hit(self) -> None:
        """Record that this pause point was hit."""
        self.hit_count += 1
        self.last_hit_time = time.time()

    def __str__(self) -> str:
        return f"PausePoint({self.name}, enabled={self.enabled}, hits={self.hit_count})"


class ExecutionPauseController:
    """Controller for managing execution pause points and behavior.

    Port of ExecutionPauseController from Qontinui framework.

    This controller manages:
    - Named pause points throughout the execution
    - Conditional pausing based on criteria
    - Pause point statistics and debugging
    - Integration with ExecutionController

    Features:
    - Register/unregister pause points dynamically
    - Enable/disable specific pause points
    - Conditional pause based on custom criteria
    - Pause point hit statistics
    - Step-through execution mode
    """

    def __init__(self, execution_controller: ExecutionController | None = None):
        """Initialize the pause controller.

        Args:
            execution_controller: Controller to use, or None for global
        """
        # TODO: Implement global execution controller singleton
        self.execution_controller = execution_controller
        self._pause_points: dict[str, PausePoint] = {}
        self._global_pause_enabled = True
        self._step_mode = False
        self._conditional_pause_handlers: list[Callable[[str, dict[str, Any]], bool]] = []
        self._lock = threading.RLock()

        # Pre-defined pause points
        self._register_default_pause_points()

        logger.info("ExecutionPauseController initialized")

    def _register_default_pause_points(self) -> None:
        """Register default pause points."""
        default_points = [
            ("action_start", "Before action execution"),
            ("action_end", "After action execution"),
            ("state_transition", "Before state transition"),
            ("error", "On error occurrence"),
            ("find_operation", "Before find operation"),
            ("screenshot", "Before taking screenshot"),
        ]

        for name, description in default_points:
            self.register_pause_point(name, description, enabled=False)

    def register_pause_point(self, name: str, description: str = "", enabled: bool = True) -> None:
        """Register a new pause point.

        Args:
            name: Unique name for the pause point
            description: Human-readable description
            enabled: Whether pause point starts enabled
        """
        with self._lock:
            if name in self._pause_points:
                logger.warning(f"Pause point '{name}' already registered, updating")

            self._pause_points[name] = PausePoint(name, description, enabled)
            logger.debug(f"Registered pause point: {name}")

    def unregister_pause_point(self, name: str) -> None:
        """Unregister a pause point.

        Args:
            name: Name of pause point to remove
        """
        with self._lock:
            if name in self._pause_points:
                del self._pause_points[name]
                logger.debug(f"Unregistered pause point: {name}")

    def enable_pause_point(self, name: str) -> None:
        """Enable a specific pause point.

        Args:
            name: Name of pause point to enable
        """
        with self._lock:
            if name in self._pause_points:
                self._pause_points[name].enabled = True
                logger.debug(f"Enabled pause point: {name}")
            else:
                logger.warning(f"Pause point '{name}' not found")

    def disable_pause_point(self, name: str) -> None:
        """Disable a specific pause point.

        Args:
            name: Name of pause point to disable
        """
        with self._lock:
            if name in self._pause_points:
                self._pause_points[name].enabled = False
                logger.debug(f"Disabled pause point: {name}")

    def enable_all_pause_points(self) -> None:
        """Enable all registered pause points."""
        with self._lock:
            for point in self._pause_points.values():
                point.enabled = True
            logger.info("Enabled all pause points")

    def disable_all_pause_points(self) -> None:
        """Disable all registered pause points."""
        with self._lock:
            for point in self._pause_points.values():
                point.enabled = False
            logger.info("Disabled all pause points")

    def set_global_pause(self, enabled: bool) -> None:
        """Enable or disable global pause checking.

        Args:
            enabled: Whether to enable global pause
        """
        with self._lock:
            self._global_pause_enabled = enabled
            logger.info(f"Global pause {'enabled' if enabled else 'disabled'}")

    def set_step_mode(self, enabled: bool) -> None:
        """Enable or disable step-through mode.

        In step mode, execution pauses at every pause point.

        Args:
            enabled: Whether to enable step mode
        """
        with self._lock:
            self._step_mode = enabled
            if enabled:
                self._global_pause_enabled = True
                logger.info("Step mode enabled - will pause at every point")
            else:
                logger.info("Step mode disabled")

    def check_pause_point(self, name: str, context: dict[str, Any] | None = None) -> None:
        """Check if execution should pause at this point.

        Args:
            name: Name of the pause point
            context: Optional context data for conditional pause
        """
        if not self._global_pause_enabled:
            return

        with self._lock:
            # Register unknown pause points automatically
            if name not in self._pause_points:
                self.register_pause_point(name, enabled=False)

            pause_point = self._pause_points[name]
            pause_point.hit()

            should_pause = False

            # Check if this specific point is enabled
            if pause_point.enabled:
                should_pause = True
                logger.info(f"Pause point '{name}' triggered")

            # Check step mode
            elif self._step_mode:
                should_pause = True
                logger.info(f"Step mode pause at '{name}'")

            # Check conditional pause handlers
            else:
                for handler in self._conditional_pause_handlers:
                    try:
                        if handler(name, context or {}):
                            should_pause = True
                            logger.info(f"Conditional pause at '{name}'")
                            break
                    except Exception as e:
                        logger.error(f"Error in conditional pause handler: {e}")

        # Pause if needed (outside lock to avoid deadlock)
        if should_pause:
            self._do_pause(name)

    def _do_pause(self, pause_point_name: str) -> None:
        """Execute the pause.

        Args:
            pause_point_name: Name of pause point that triggered pause
        """
        logger.info(f"Pausing at '{pause_point_name}'")

        # Only pause if we have a controller and we're running
        if self.execution_controller is None:
            logger.warning("No execution controller available, cannot pause")
            return

        if self.execution_controller.is_running():
            self.execution_controller.pause()

            # Wait for resume
            # TODO: Implement await_not_paused method in ExecutionController protocol
            # self.execution_controller.await_not_paused()
            # For now, use a simple polling loop
            while self.execution_controller.is_paused():
                time.sleep(0.1)

            # In step mode, prepare for next pause
            if self._step_mode:
                logger.debug("Step mode - ready for next step")

    def add_conditional_pause_handler(self, handler: Callable[[str, dict[str, Any]], bool]) -> None:
        """Add a conditional pause handler.

        Handler receives pause point name and context, returns True to pause.

        Args:
            handler: Function to evaluate pause condition
        """
        with self._lock:
            self._conditional_pause_handlers.append(handler)

    def remove_conditional_pause_handler(
        self, handler: Callable[[str, dict[str, Any]], bool]
    ) -> None:
        """Remove a conditional pause handler.

        Args:
            handler: Handler to remove
        """
        with self._lock:
            if handler in self._conditional_pause_handlers:
                self._conditional_pause_handlers.remove(handler)

    def get_pause_points(self) -> dict[str, PausePoint]:
        """Get all registered pause points.

        Returns:
            Dictionary of pause point names to PausePoint objects
        """
        with self._lock:
            return dict(self._pause_points)

    def get_statistics(self) -> dict[str, dict[str, Any]]:
        """Get pause point statistics.

        Returns:
            Dictionary of statistics per pause point
        """
        with self._lock:
            stats = {}
            for name, point in self._pause_points.items():
                stats[name] = {
                    "enabled": point.enabled,
                    "hit_count": point.hit_count,
                    "last_hit_time": point.last_hit_time,
                    "description": point.description,
                }
            return stats

    def reset_statistics(self) -> None:
        """Reset all pause point statistics."""
        with self._lock:
            for point in self._pause_points.values():
                point.hit_count = 0
                point.last_hit_time = None
            logger.info("Pause point statistics reset")


# Global pause controller instance
_global_pause_controller: ExecutionPauseController | None = None


def get_pause_controller() -> ExecutionPauseController:
    """Get the global pause controller.

    Returns:
        The global ExecutionPauseController instance
    """
    global _global_pause_controller
    if _global_pause_controller is None:
        _global_pause_controller = ExecutionPauseController()
    return _global_pause_controller


def check_pause(name: str, context: dict[str, Any] | None = None) -> None:
    """Check if execution should pause at this point.

    Args:
        name: Name of the pause point
        context: Optional context data
    """
    get_pause_controller().check_pause_point(name, context)


def enable_step_mode() -> None:
    """Enable step-through execution mode."""
    get_pause_controller().set_step_mode(True)


def disable_step_mode() -> None:
    """Disable step-through execution mode."""
    get_pause_controller().set_step_mode(False)
