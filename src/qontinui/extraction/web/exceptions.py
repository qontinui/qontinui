"""
Exception types and retry utilities for web extraction.

Provides:
- Specific exception types for different failure modes
- Retry decorator for transient failures
- Timeout handling utilities
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# Type variable for async functions
F = TypeVar("F", bound=Callable[..., Any])


class WebExtractionError(Exception):
    """Base exception for web extraction errors."""

    pass


class ExtractionTimeoutError(WebExtractionError):
    """Raised when an extraction operation times out."""

    pass


class CDPError(WebExtractionError):
    """Raised when a Chrome DevTools Protocol operation fails."""

    pass


class FrameExtractionError(WebExtractionError):
    """Raised when extracting from a frame fails."""

    pass


class ShadowDOMError(WebExtractionError):
    """Raised when shadow DOM extraction fails."""

    pass


class ElementExtractionError(WebExtractionError):
    """Raised when element extraction fails."""

    pass


class ValidationError(WebExtractionError):
    """Raised when input validation fails."""

    pass


def with_retry(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    exponential_backoff: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator to add retry logic to async functions.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        exponential_backoff: Whether to use exponential backoff.
        retryable_exceptions: Tuple of exception types to retry on.

    Returns:
        Decorated function with retry logic.

    Example:
        @with_retry(max_retries=3, base_delay=0.5)
        async def fetch_data():
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
                        raise

                    # Calculate delay
                    if exponential_backoff:
                        delay = min(base_delay * (2**attempt), max_delay)
                    else:
                        delay = base_delay

                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)

            # Should never reach here, but satisfy type checker
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry loop exit")

        return wrapper  # type: ignore[return-value]

    return decorator


async def with_timeout(
    coro: Any,
    timeout_seconds: float,
    operation_name: str = "operation",
) -> Any:
    """
    Execute a coroutine with a timeout.

    Args:
        coro: Coroutine to execute.
        timeout_seconds: Timeout in seconds.
        operation_name: Name for error messages.

    Returns:
        Result of the coroutine.

    Raises:
        ExtractionTimeoutError: If the operation times out.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except TimeoutError:
        raise ExtractionTimeoutError(
            f"{operation_name} timed out after {timeout_seconds}s"
        )
