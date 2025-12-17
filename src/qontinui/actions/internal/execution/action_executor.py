"""ActionExecutor - ported from Qontinui framework.

Centralized action execution with lifecycle management.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

from .....model.action.action_history import ActionHistory
from .....model.action.action_record import ActionRecordBuilder
from ....action_config import ActionConfig
from ....action_interface import ActionInterface

logger = logging.getLogger(__name__)


class ExecutionPhase(Enum):
    """Phases of action execution."""

    INITIALIZING = auto()
    PRE_EXECUTION = auto()
    EXECUTING = auto()
    POST_EXECUTION = auto()
    COMPLETED = auto()
    FAILED = auto()


class ExecutionContext:
    """Context for action execution."""

    def __init__(self, action: ActionInterface, target: Any | None = None) -> None:
        """Initialize execution context.

        Args:
            action: Action to execute
            target: Target for action
        """
        self.action = action
        self.target = target
        self.phase = ExecutionPhase.INITIALIZING
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.result: bool | None = None
        self.error: Exception | None = None
        self.metadata: dict[str, Any] = {}
        self.thread_id = threading.current_thread().ident

    @property
    def duration(self) -> float:
        """Get execution duration.

        Returns:
            Duration in seconds
        """
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def is_complete(self) -> bool:
        """Check if execution is complete.

        Returns:
            True if complete
        """
        return self.phase in (ExecutionPhase.COMPLETED, ExecutionPhase.FAILED)

    @property
    def is_successful(self) -> bool:
        """Check if execution was successful.

        Returns:
            True if successful
        """
        return self.phase == ExecutionPhase.COMPLETED and (self.result is True)


@dataclass
class ExecutorConfig:
    """Configuration for ActionExecutor."""

    enable_logging: bool = True
    enable_history: bool = True
    enable_metrics: bool = True
    max_history_size: int = 1000
    timeout_seconds: float = 30.0
    retry_on_failure: bool = False
    max_retries: int = 3
    retry_delay: float = 1.0
    pre_execution_hooks: list[Callable[[ExecutionContext], None]] = field(
        default_factory=list
    )
    post_execution_hooks: list[Callable[[ExecutionContext], None]] = field(
        default_factory=list
    )


class ActionExecutor:
    """Centralized action executor with lifecycle management.

    Port of ActionExecutor from Qontinui framework.

    Provides centralized execution of all actions with:
    - Lifecycle management
    - Execution hooks
    - History tracking
    - Metrics collection
    - Error handling
    - Retry logic
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for executor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize executor."""
        if not hasattr(self, "_initialized"):
            self.config = ExecutorConfig()
            self._history = ActionHistory()
            self._current_context: ExecutionContext | None = None
            self._execution_count = 0
            self._success_count = 0
            self._failure_count = 0
            self._total_duration = 0.0
            self._initialized = True
            logger.info("ActionExecutor initialized")

    def configure(self, config: ExecutorConfig) -> "ActionExecutor":
        """Configure the executor.

        Args:
            config: Executor configuration

        Returns:
            Self for fluent interface
        """
        self.config = config
        return self

    def execute(self, action: ActionInterface, target: Any | None = None) -> bool:
        """Execute an action with full lifecycle management.

        Args:
            action: Action to execute
            target: Optional target

        Returns:
            True if successful
        """
        context = ExecutionContext(action, target)
        self._current_context = context

        try:
            # Initialize
            self._initialize_execution(context)

            # Pre-execution
            self._pre_execution(context)

            # Execute with retry logic
            if self.config.retry_on_failure:
                result = self._execute_with_retry(context)
            else:
                result = self._execute_action(context)

            context.result = result

            # Post-execution
            self._post_execution(context)

            # Complete
            self._complete_execution(context)

            return result

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            context.error = e
            context.phase = ExecutionPhase.FAILED
            self._handle_failure(context)
            return False

        finally:
            self._current_context = None

    def _initialize_execution(self, context: ExecutionContext) -> None:
        """Initialize execution phase.

        Args:
            context: Execution context
        """
        context.phase = ExecutionPhase.INITIALIZING
        context.start_time = time.time()
        self._execution_count += 1

        if self.config.enable_logging:
            logger.debug(
                f"Initializing action execution: {context.action.__class__.__name__}"
            )

    def _pre_execution(self, context: ExecutionContext) -> None:
        """Pre-execution phase.

        Args:
            context: Execution context
        """
        context.phase = ExecutionPhase.PRE_EXECUTION

        # Run pre-execution hooks
        for hook in self.config.pre_execution_hooks:
            try:
                hook(context)
            except Exception as e:
                logger.warning(f"Pre-execution hook failed: {e}")

        # Apply pre-action pause if configured
        if hasattr(context.action, "options"):
            options = context.action.options
            if isinstance(options, ActionConfig) and options.pause_before > 0:
                time.sleep(options.pause_before)

    def _execute_action(self, context: ExecutionContext) -> bool:
        """Execute the actual action.

        Args:
            context: Execution context

        Returns:
            True if successful
        """
        context.phase = ExecutionPhase.EXECUTING

        try:
            if context.target is not None:
                if hasattr(context.action, "execute"):
                    result = context.action.execute(context.target)
                else:
                    result = context.action(context.target)
            else:
                if hasattr(context.action, "execute"):
                    result = context.action.execute()
                else:
                    result = context.action()

            return bool(result)

        except Exception as e:
            logger.error(f"Action execution error: {e}")
            context.error = e
            return False

    def _execute_with_retry(self, context: ExecutionContext) -> bool:
        """Execute with retry logic.

        Args:
            context: Execution context

        Returns:
            True if eventually successful
        """
        for attempt in range(self.config.max_retries + 1):
            if attempt > 0:
                logger.info(f"Retry attempt {attempt}/{self.config.max_retries}")
                time.sleep(self.config.retry_delay)

            result = self._execute_action(context)
            if result:
                return True

        return False

    def _post_execution(self, context: ExecutionContext) -> None:
        """Post-execution phase.

        Args:
            context: Execution context
        """
        context.phase = ExecutionPhase.POST_EXECUTION
        context.end_time = time.time()

        # Apply post-action pause if configured
        if hasattr(context.action, "options"):
            options = context.action.options
            if isinstance(options, ActionConfig) and options.pause_after > 0:
                time.sleep(options.pause_after)

        # Run post-execution hooks
        for hook in self.config.post_execution_hooks:
            try:
                hook(context)
            except Exception as e:
                logger.warning(f"Post-execution hook failed: {e}")

    def _complete_execution(self, context: ExecutionContext) -> None:
        """Complete execution phase.

        Args:
            context: Execution context
        """
        if context.result:
            context.phase = ExecutionPhase.COMPLETED
            self._success_count += 1
        else:
            context.phase = ExecutionPhase.FAILED
            self._failure_count += 1

        self._total_duration += context.duration

        # Record in history
        if self.config.enable_history:
            self._record_execution(context)

        # Log completion
        if self.config.enable_logging:
            status = "successful" if context.result else "failed"
            logger.info(f"Action {status} in {context.duration:.3f}s")

    def _handle_failure(self, context: ExecutionContext) -> None:
        """Handle execution failure.

        Args:
            context: Execution context
        """
        self._failure_count += 1

        if self.config.enable_logging:
            logger.error(f"Action failed: {context.error}")

        if self.config.enable_history:
            self._record_execution(context)

    def _record_execution(self, context: ExecutionContext) -> None:
        """Record execution in history.

        Args:
            context: Execution context
        """
        builder = ActionRecordBuilder()

        if hasattr(context.action, "options"):
            builder.with_action_config(context.action.options)

        timestamp = (
            datetime.fromtimestamp(context.start_time)
            if context.start_time is not None
            else datetime.now()
        )
        record = (
            builder.with_text(context.action.__class__.__name__)
            .with_duration(context.duration)
            .with_success(context.result or False)
            .with_timestamp(timestamp)
            .build()
        )

        self._history.add_record(record)

        # Trim history if needed
        while self._history.size() > self.config.max_history_size:
            self._history.remove_oldest()

    def add_pre_execution_hook(self, hook: Callable[[ExecutionContext], None]) -> None:
        """Add pre-execution hook.

        Args:
            hook: Hook function
        """
        self.config.pre_execution_hooks.append(hook)

    def add_post_execution_hook(self, hook: Callable[[ExecutionContext], None]) -> None:
        """Add post-execution hook.

        Args:
            hook: Hook function
        """
        self.config.post_execution_hooks.append(hook)

    def get_history(self) -> ActionHistory:
        """Get execution history.

        Returns:
            Action history
        """
        return self._history

    def get_metrics(self) -> dict[str, Any]:
        """Get execution metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "total_executions": self._execution_count,
            "successful": self._success_count,
            "failed": self._failure_count,
            "success_rate": self._success_count / max(1, self._execution_count) * 100,
            "average_duration": self._total_duration / max(1, self._execution_count),
            "total_duration": self._total_duration,
        }

    def get_current_context(self) -> ExecutionContext | None:
        """Get current execution context.

        Returns:
            Current context or None
        """
        return self._current_context

    def reset_metrics(self) -> None:
        """Reset execution metrics."""
        self._execution_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._total_duration = 0.0
        logger.info("Metrics reset")

    def clear_history(self) -> None:
        """Clear execution history."""
        self._history.clear()
        logger.info("History cleared")

    @classmethod
    def get_instance(cls) -> "ActionExecutor":
        """Get singleton instance.

        Returns:
            ActionExecutor instance
        """
        return cls()


# Global executor instance
executor = ActionExecutor.get_instance()
