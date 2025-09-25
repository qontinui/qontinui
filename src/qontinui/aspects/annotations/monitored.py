"""Monitored annotation - ported from Qontinui framework.

Marks methods or classes for enhanced performance monitoring.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MonitoredConfig:
    """Configuration for @monitored decorator."""

    name: str = ""
    """Custom name for the monitored operation."""

    threshold: int = -1
    """Performance threshold in milliseconds."""

    track_memory: bool = False
    """Whether to track memory usage."""

    log_parameters: bool = False
    """Whether to include method parameters in logs."""

    log_result: bool = False
    """Whether to include return value in logs."""

    tags: list[str] = None
    """Tags for categorizing the operation."""

    sampling_rate: float = 1.0
    """Sampling rate (0.0 to 1.0)."""

    create_span: bool = False
    """Whether to create a trace span."""

    custom_metrics: list[str] = None
    """Custom metrics to capture."""

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.custom_metrics is None:
            self.custom_metrics = []


def monitored(
    name: str = "",
    threshold: int = -1,
    track_memory: bool = False,
    log_parameters: bool = False,
    log_result: bool = False,
    tags: list[str] | None = None,
    sampling_rate: float = 1.0,
    create_span: bool = False,
    custom_metrics: list[str] | None = None,
) -> Callable:
    """Mark a method or class for enhanced performance monitoring.

    Direct port of Brobot's @Monitored annotation.

    Methods decorated with @monitored will have detailed performance
    metrics collected, including custom thresholds and alerting.

    Example usage:
        @monitored(threshold=5000, track_memory=True)
        def perform_expensive_operation():
            # Operation that should complete within 5 seconds
            pass

        @monitored(name="UserLogin", tags=["authentication", "critical"])
        def login_user(username: str, password: str):
            # Custom name and tags for categorization
            pass

    Args:
        name: Custom name for the monitored operation.
             Default is the method signature.
        threshold: Performance threshold in milliseconds.
                  Operations exceeding this will trigger alerts.
                  Default is -1 (use global threshold).
        track_memory: Whether to track memory usage for this operation.
                     Default is False.
        log_parameters: Whether to include method parameters in logs.
                       Default is False for security.
        log_result: Whether to include return value in logs.
                   Default is False for security.
        tags: Tags for categorizing the operation.
             Useful for filtering and reporting.
        sampling_rate: Sampling rate for this operation (0.0 to 1.0).
                      1.0 means monitor every call, 0.1 means monitor 10% of calls.
                      Default is 1.0.
        create_span: Whether to create a trace span for distributed tracing.
                    Default is False.
        custom_metrics: Custom metrics to capture.

    Returns:
        Decorator function or decorated class
    """
    config = MonitoredConfig(
        name=name,
        threshold=threshold,
        track_memory=track_memory,
        log_parameters=log_parameters,
        log_result=log_result,
        tags=tags or [],
        sampling_rate=sampling_rate,
        create_span=create_span,
        custom_metrics=custom_metrics or [],
    )

    def decorator(func_or_class):
        if isinstance(func_or_class, type):
            # Class decorator
            return _decorate_class(func_or_class, config)
        else:
            # Function decorator
            return _decorate_function(func_or_class, config)

    return decorator


def _decorate_function(func: Callable, config: MonitoredConfig) -> Callable:
    """Decorate a function with monitoring.

    Args:
        func: Function to decorate
        config: Monitoring configuration

    Returns:
        Decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check sampling rate
        import random

        if random.random() > config.sampling_rate:
            return func(*args, **kwargs)

        # Start monitoring
        start_time = time.time()
        operation_name = config.name or func.__name__

        if config.log_parameters:
            logger.debug(f"Starting {operation_name} with args={args}, kwargs={kwargs}")
        else:
            logger.debug(f"Starting {operation_name}")

        # Track memory if requested
        start_memory = None
        if config.track_memory:
            import psutil

            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB

        try:
            # Execute the function
            result = func(*args, **kwargs)

            # Calculate metrics
            elapsed_ms = (time.time() - start_time) * 1000

            # Check threshold
            if config.threshold > 0 and elapsed_ms > config.threshold:
                logger.warning(
                    f"{operation_name} exceeded threshold: {elapsed_ms:.2f}ms > {config.threshold}ms"
                )

            # Log result if requested
            if config.log_result:
                logger.debug(
                    f"{operation_name} completed in {elapsed_ms:.2f}ms with result: {result}"
                )
            else:
                logger.debug(f"{operation_name} completed in {elapsed_ms:.2f}ms")

            # Track memory delta
            if config.track_memory and start_memory is not None:
                import psutil

                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = end_memory - start_memory
                logger.debug(f"{operation_name} memory delta: {memory_delta:.2f}MB")

            return result

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"{operation_name} failed after {elapsed_ms:.2f}ms", exc_info=e)
            raise

    # Store config on the wrapper
    wrapper._monitored_config = config
    wrapper._monitored = True

    return wrapper


def _decorate_class(cls: type, config: MonitoredConfig) -> type:
    """Decorate a class with monitoring.

    Args:
        cls: Class to decorate
        config: Monitoring configuration

    Returns:
        Decorated class
    """
    # Apply monitoring to all public methods
    for attr_name in dir(cls):
        if not attr_name.startswith("_"):
            attr = getattr(cls, attr_name)
            if callable(attr) and not isinstance(attr, type):
                # Create a copy of config with method-specific name
                method_config = MonitoredConfig(
                    name=f"{cls.__name__}.{attr_name}",
                    threshold=config.threshold,
                    track_memory=config.track_memory,
                    log_parameters=config.log_parameters,
                    log_result=config.log_result,
                    tags=config.tags,
                    sampling_rate=config.sampling_rate,
                    create_span=config.create_span,
                    custom_metrics=config.custom_metrics,
                )
                setattr(cls, attr_name, _decorate_function(attr, method_config))

    # Mark class as monitored
    cls._monitored = True
    cls._monitored_config = config

    return cls


def is_monitored(obj: Any) -> bool:
    """Check if an object is monitored.

    Args:
        obj: Object to check

    Returns:
        True if object is decorated with @monitored
    """
    return hasattr(obj, "_monitored") and obj._monitored


def get_monitored_config(obj: Any) -> MonitoredConfig | None:
    """Get monitoring configuration from an object.

    Args:
        obj: Monitored object

    Returns:
        Monitoring configuration or None
    """
    if not is_monitored(obj):
        return None

    return getattr(obj, "_monitored_config", None)
