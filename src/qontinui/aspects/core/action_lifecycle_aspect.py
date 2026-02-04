"""Action lifecycle aspect - ported from Qontinui framework.

Manages the complete lifecycle of action executions.
"""

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from threading import local
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...actions.action_interface import ActionInterface
    from ...actions.action_result import ActionResult
    from ...actions.object_collection import ObjectCollection

logger = logging.getLogger(__name__)


@dataclass
class ActionContext:
    """Context information for the current action execution."""

    action_id: str
    """Unique ID for this action execution."""

    action_type: str
    """Type of action being executed."""

    action_class: str
    """Class name of the action."""

    start_time: float
    """Start timestamp."""

    end_time: float | None = None
    """End timestamp."""

    success: bool = False
    """Whether action succeeded."""

    error: Exception | None = None
    """Error if action failed."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    @property
    def duration_ms(self) -> float | None:
        """Get duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000


class ActionLifecycleAspect:
    """Manages the lifecycle of all action executions.

    Port of ActionLifecycleAspect from Qontinui framework.

    This aspect centralizes cross-cutting concerns that were previously
    scattered throughout the ActionExecution class:
    - Pre-execution setup (timing, logging, pause points)
    - Post-execution tasks (screenshots, metrics, dataset collection)
    - Execution controller pause points
    - Automatic retry logic for transient failures

    By extracting these concerns into an aspect, the core action logic
    becomes cleaner and more focused on its primary responsibility.
    """

    def __init__(
        self,
        pre_action_pause: float = 0.0,
        post_action_pause: float = 0.0,
        log_events: bool = True,
        capture_before_screenshot: bool = False,
        capture_after_screenshot: bool = True,
    ) -> None:
        """Initialize the aspect.

        Args:
            pre_action_pause: Seconds to pause before action
            post_action_pause: Seconds to pause after action
            log_events: Whether to log lifecycle events
            capture_before_screenshot: Capture screenshot before action
            capture_after_screenshot: Capture screenshot after action
        """
        self.pre_action_pause = pre_action_pause
        self.post_action_pause = post_action_pause
        self.log_events = log_events
        self.capture_before_screenshot = capture_before_screenshot
        self.capture_after_screenshot = capture_after_screenshot

        # Thread-local storage for action context
        self._local = local()

        # Global metrics
        self._total_actions = 0
        self._successful_actions = 0
        self._failed_actions = 0
        self._action_history: list[ActionContext] = []

    def get_current_context(self) -> ActionContext | None:
        """Get the current action context for this thread.

        Returns:
            Current action context or None
        """
        return getattr(self._local, "context", None)

    def wrap_action(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap an action's perform method with lifecycle management.

        Args:
            func: The perform method to wrap

        Returns:
            Wrapped function
        """

        @wraps(func)
        def wrapper(
            action_instance: "ActionInterface",
            action_result: "ActionResult",
            *object_collections: "ObjectCollection",
        ) -> Any:
            # Create action context
            context = ActionContext(
                action_id=str(uuid.uuid4()),
                action_type=action_instance.get_action_type().name,
                action_class=action_instance.__class__.__name__,
                start_time=time.time(),
            )

            # Store in thread-local
            self._local.context = context

            try:
                # Pre-execution
                self._pre_execution(context, action_instance, action_result, object_collections)

                # Execute action
                result = func(action_instance, action_result, *object_collections)

                # Mark success
                context.success = True

                # Post-execution
                self._post_execution(context, action_instance, action_result, True)

                return result

            except Exception as e:
                # Mark failure
                context.success = False
                context.error = e

                # Post-execution with failure
                self._post_execution(context, action_instance, action_result, False)

                # Re-raise
                raise

            finally:
                # Clear thread-local context
                self._local.context = None

                # Update global metrics
                self._update_metrics(context)

        return wrapper

    def _pre_execution(
        self,
        context: ActionContext,
        action: "ActionInterface",
        action_result: "ActionResult",
        object_collections: tuple[Any, ...],
    ) -> None:
        """Handle pre-execution tasks.

        Args:
            context: Action context
            action: Action being executed
            action_result: Result accumulator
            object_collections: Object collections
        """
        if self.log_events:
            logger.debug(
                f"[{context.action_id}] Starting {context.action_type} ({context.action_class})"
            )

        # Pre-action pause
        if self.pre_action_pause > 0:
            logger.debug(f"Pre-action pause: {self.pre_action_pause}s")
            time.sleep(self.pre_action_pause)

        # Capture before screenshot
        if self.capture_before_screenshot:
            self._capture_screenshot(context, "before")

    def _post_execution(
        self,
        context: ActionContext,
        action: "ActionInterface",
        action_result: "ActionResult",
        success: bool,
    ) -> None:
        """Handle post-execution tasks.

        Args:
            context: Action context
            action: Action that was executed
            action_result: Result accumulator
            success: Whether action succeeded
        """
        # Record end time
        context.end_time = time.time()

        # Log completion
        if self.log_events:
            if success:
                logger.debug(
                    f"[{context.action_id}] Completed {context.action_type} "
                    f"in {context.duration_ms:.2f}ms"
                )
            else:
                logger.error(
                    f"[{context.action_id}] Failed {context.action_type} "
                    f"after {context.duration_ms:.2f}ms: {context.error}"
                )

        # Capture after screenshot
        if self.capture_after_screenshot:
            self._capture_screenshot(context, "after")

        # Post-action pause
        if self.post_action_pause > 0:
            logger.debug(f"Post-action pause: {self.post_action_pause}s")
            time.sleep(self.post_action_pause)

    def _capture_screenshot(self, context: ActionContext, phase: str) -> None:
        """Capture a screenshot.

        Args:
            context: Action context
            phase: Phase of capture (before/after)
        """
        # This would integrate with screenshot capture service
        logger.debug(
            f"Would capture {phase} screenshot for {context.action_type} [{context.action_id}]"
        )

    def _update_metrics(self, context: ActionContext) -> None:
        """Update global metrics.

        Args:
            context: Completed action context
        """
        self._total_actions += 1

        if context.success:
            self._successful_actions += 1
        else:
            self._failed_actions += 1

        # Keep limited history
        self._action_history.append(context)
        if len(self._action_history) > 1000:
            self._action_history.pop(0)

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics.

        Returns:
            Dictionary of metrics
        """
        success_rate = 0.0
        if self._total_actions > 0:
            success_rate = self._successful_actions / self._total_actions * 100

        return {
            "total_actions": self._total_actions,
            "successful_actions": self._successful_actions,
            "failed_actions": self._failed_actions,
            "success_rate": success_rate,
            "history_size": len(self._action_history),
        }

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._total_actions = 0
        self._successful_actions = 0
        self._failed_actions = 0
        self._action_history.clear()

    def get_action_history(
        self, action_type: str | None = None, limit: int = 100
    ) -> list[ActionContext]:
        """Get action execution history.

        Args:
            action_type: Filter by action type
            limit: Maximum number of entries

        Returns:
            List of action contexts
        """
        history = self._action_history

        if action_type:
            history = [c for c in history if c.action_type == action_type]

        return history[-limit:]


# Global instance
_lifecycle_aspect = ActionLifecycleAspect()


def with_lifecycle(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to apply lifecycle management to an action.

    Usage:
        @with_lifecycle
        def perform(self, matches, *collections):
            # action implementation

    Args:
        func: Function to wrap

    Returns:
        Wrapped function
    """
    return _lifecycle_aspect.wrap_action(func)


def get_lifecycle_aspect() -> ActionLifecycleAspect:
    """Get the global lifecycle aspect instance.

    Returns:
        The lifecycle aspect
    """
    return _lifecycle_aspect
