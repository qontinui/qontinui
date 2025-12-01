"""Structured logging configuration for Qontinui using structlog.

This replaces Brobot's custom logging framework with a modern Python solution
that provides structured logging, context preservation, and performance optimization.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import structlog

from ..config import get_settings


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    structured: bool = True,
    console: bool = True,
    add_timestamp: bool = True,
    add_caller_info: bool = True,
    colorize: bool = True,
) -> None:
    """Configure structured logging for Qontinui.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        structured: Use JSON structured output
        console: Enable console output (overridden by QONTINUI_DISABLE_CONSOLE_LOGGING env var)
        add_timestamp: Add timestamps to logs
        add_caller_info: Add caller information
        colorize: Colorize console output (only for non-structured)
    """
    import os

    # CRITICAL: Check environment variable to disable console logging
    # When running under Rust executor, this prevents JSON parse errors
    if os.getenv("QONTINUI_DISABLE_CONSOLE_LOGGING") == "1":
        console = False
        log_file = None  # Also disable file logging to minimize I/O
    # Build processor chain
    processors: list[Any] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
    ]

    if add_timestamp:
        processors.append(structlog.processors.TimeStamper(fmt="iso"))

    if add_caller_info:
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                ]
            )
        )

    processors.extend(
        [
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
    )

    # Add final renderer
    if structured:
        processors.append(structlog.processors.JSONRenderer())
    else:
        if colorize and console:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=False))

    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    handlers: list[logging.Handler] = []

    if console:
        # IMPORTANT: Use stderr for logs so stdout is reserved for JSON messages to Rust
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(file_handler)

    # If no handlers (console logging disabled), use NullHandler to completely suppress output
    if not handlers:
        handlers.append(logging.NullHandler())
        level = "CRITICAL"  # Set to highest level to minimize overhead

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
        handlers=handlers,
        force=True,
    )


# Global state for lazy initialization
_logging_initialized = False


def _ensure_logging_initialized() -> None:
    """Ensure logging is initialized (called lazily, not at import time)."""
    global _logging_initialized

    if _logging_initialized:
        return

    # CRITICAL: Check environment variable FIRST before any logging setup
    # When running under Rust executor, completely disable all logging
    import os

    if os.getenv("QONTINUI_DISABLE_CONSOLE_LOGGING") == "1":
        # Set to disabled state without calling setup_logging at all
        logging.disable(logging.CRITICAL)  # Disable ALL logging output
        _logging_initialized = True
        return

    # Initialize logging with defaults
    try:
        settings = get_settings()
        if hasattr(settings, "log_path") and hasattr(settings, "debug_mode"):
            log_file = (
                settings.log_path / f"qontinui_{datetime.now().strftime('%Y%m%d')}.log"
            )
            setup_logging(
                level="DEBUG" if settings.debug_mode else "INFO",
                log_file=log_file,
                structured=not settings.debug_mode,  # Use readable format in debug mode
                colorize=settings.debug_mode,
            )
    except (ImportError, AttributeError, OSError, ValueError):
        # If settings fail or log path is invalid, just use basic logging
        setup_logging(level="INFO", structured=False)

    _logging_initialized = True


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Structured logger instance
    """
    # Ensure logging is initialized before creating logger
    _ensure_logging_initialized()
    return cast(structlog.BoundLogger, structlog.get_logger(name))


class LogContext:
    """Context manager for temporary log context."""

    def __init__(self, logger: structlog.BoundLogger, **kwargs) -> None:
        """Initialize with logger and context.

        Args:
            logger: Logger instance
            **kwargs: Context key-value pairs
        """
        self.logger = logger
        self.context = kwargs
        self.original_context: dict[str, Any] = {}

    def __enter__(self) -> structlog.types.BindableLogger:
        """Enter context and bind values."""
        self.bound_logger = self.logger.bind(**self.context)
        return self.bound_logger

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and unbind values."""
        # Context automatically cleaned up when bound_logger goes out of scope
        pass


class ActionLogger:
    """Specialized logger for action execution."""

    def __init__(self, base_logger: structlog.BoundLogger | None = None) -> None:
        """Initialize action logger.

        Args:
            base_logger: Base logger to use
        """
        self.logger = base_logger or get_logger(__name__)

    def log_action_start(
        self, action_type: str, target: Any, **kwargs
    ) -> dict[str, Any]:
        """Log action start.

        Args:
            action_type: Type of action
            target: Action target
            **kwargs: Additional context

        Returns:
            Action context dict
        """
        context = {
            "action_type": action_type,
            "target": str(target),
            "start_time": datetime.utcnow().isoformat(),
            **kwargs,
        }

        self.logger.info("action_started", **context)

        return context

    def log_action_end(
        self,
        context: dict[str, Any],
        success: bool,
        result: Any = None,
        error: Exception | None = None,
    ) -> None:
        """Log action end.

        Args:
            context: Action context from log_action_start
            success: Whether action succeeded
            result: Action result
            error: Optional error
        """
        end_time = datetime.utcnow()
        start_time = datetime.fromisoformat(context["start_time"])
        duration = (end_time - start_time).total_seconds()

        log_data = {
            **context,
            "end_time": end_time.isoformat(),
            "duration": duration,
            "success": success,
        }

        if result is not None:
            log_data["result"] = str(result)

        if error:
            log_data["error"] = str(error)
            log_data["error_type"] = type(error).__name__

        if success:
            self.logger.info("action_completed", **log_data)
        else:
            self.logger.error("action_failed", **log_data)


class StateLogger:
    """Specialized logger for state transitions."""

    def __init__(self, base_logger: structlog.BoundLogger | None = None) -> None:
        """Initialize state logger.

        Args:
            base_logger: Base logger to use
        """
        self.logger = base_logger or get_logger(__name__)

    def log_transition(
        self,
        from_state: str,
        to_state: str,
        trigger: str | None = None,
        success: bool = True,
        **kwargs,
    ) -> None:
        """Log state transition.

        Args:
            from_state: Source state
            to_state: Target state
            trigger: Transition trigger
            success: Whether transition succeeded
            **kwargs: Additional context
        """
        log_data = {
            "from_state": from_state,
            "to_state": to_state,
            "success": success,
            **kwargs,
        }

        if trigger:
            log_data["trigger"] = trigger

        if success:
            self.logger.info("state_transition", **log_data)
        else:
            self.logger.warning("state_transition_failed", **log_data)

    def log_state_activation(
        self, state: str, confidence: float, method: str = "unknown", **kwargs
    ) -> None:
        """Log state activation.

        Args:
            state: State name
            confidence: Activation confidence
            method: Detection method
            **kwargs: Additional context
        """
        self.logger.info(
            "state_activated",
            state=state,
            confidence=confidence,
            method=method,
            **kwargs,
        )


class PerformanceLogger:
    """Logger for performance metrics."""

    def __init__(self, base_logger: structlog.BoundLogger | None = None) -> None:
        """Initialize performance logger.

        Args:
            base_logger: Base logger to use
        """
        self.logger = base_logger or get_logger(__name__)
        self.metrics: dict[str, list[float]] = {}

    def log_timing(self, operation: str, duration: float, **kwargs) -> None:
        """Log operation timing.

        Args:
            operation: Operation name
            duration: Duration in seconds
            **kwargs: Additional context
        """
        # Store metric
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)

        # Log
        self.logger.debug(
            "performance_timing", operation=operation, duration=duration, **kwargs
        )

    def log_resource_usage(
        self, cpu_percent: float, memory_mb: float, **kwargs
    ) -> None:
        """Log resource usage.

        Args:
            cpu_percent: CPU usage percentage
            memory_mb: Memory usage in MB
            **kwargs: Additional context
        """
        self.logger.debug(
            "resource_usage", cpu_percent=cpu_percent, memory_mb=memory_mb, **kwargs
        )

    def get_stats(self, operation: str | None = None) -> dict[str, Any]:
        """Get performance statistics.

        Args:
            operation: Optional specific operation

        Returns:
            Statistics dict
        """
        if operation:
            if operation not in self.metrics:
                return {}

            values = self.metrics[operation]
            return {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "total": sum(values),
            }
        else:
            stats = {}
            for op, values in self.metrics.items():
                if values:
                    stats[op] = {
                        "count": len(values),
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "total": sum(values),
                    }
            return stats


# Create default loggers (lazy initialization - only created when first accessed)
action_logger: ActionLogger | None = None
state_logger: StateLogger | None = None
performance_logger: PerformanceLogger | None = None
