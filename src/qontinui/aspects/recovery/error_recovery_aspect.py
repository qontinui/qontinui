"""Error recovery aspect - ported from Qontinui framework.

Provides automatic error recovery with configurable retry policies.
"""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types."""

    FIXED_DELAY = "fixed_delay"
    """Fixed delay between retries."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    """Exponential backoff between retries."""

    LINEAR_BACKOFF = "linear_backoff"
    """Linear increase in delay between retries."""

    FIBONACCI_BACKOFF = "fibonacci_backoff"
    """Fibonacci sequence for delays."""


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    """Maximum number of retry attempts."""

    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    """Retry strategy to use."""

    initial_delay_ms: float = 1000
    """Initial delay in milliseconds."""

    max_delay_ms: float = 30000
    """Maximum delay in milliseconds."""

    backoff_multiplier: float = 2.0
    """Multiplier for exponential backoff."""

    linear_increment_ms: float = 1000
    """Increment for linear backoff."""

    recoverable_exceptions: list[type[Exception]] | None = None
    """List of exceptions to recover from. None means all."""

    non_recoverable_exceptions: list[type[Exception]] | None = None
    """List of exceptions to not recover from."""

    def should_retry(self, exception: Exception) -> bool:
        """Check if exception should trigger retry.

        Args:
            exception: The exception that occurred

        Returns:
            True if should retry
        """
        # Check non-recoverable first
        if self.non_recoverable_exceptions:
            for exc_type in self.non_recoverable_exceptions:
                if isinstance(exception, exc_type):
                    return False

        # Check recoverable
        if self.recoverable_exceptions:
            for exc_type in self.recoverable_exceptions:
                if isinstance(exception, exc_type):
                    return True
            return False

        # Default to retry all exceptions
        return True

    def get_delay(self, attempt: int) -> float:
        """Get delay for given attempt number.

        Args:
            attempt: Attempt number (0-based)

        Returns:
            Delay in milliseconds
        """
        if self.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.initial_delay_ms

        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.initial_delay_ms * (self.backoff_multiplier**attempt)

        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.initial_delay_ms + (self.linear_increment_ms * attempt)

        elif self.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = self._fibonacci_delay(attempt)

        else:
            delay = self.initial_delay_ms

        return min(delay, self.max_delay_ms)

    def _fibonacci_delay(self, n: int) -> float:
        """Calculate Fibonacci delay.

        Args:
            n: Sequence number

        Returns:
            Delay in milliseconds
        """
        if n <= 0:
            return self.initial_delay_ms
        elif n == 1:
            return self.initial_delay_ms * 2

        a, b = self.initial_delay_ms, self.initial_delay_ms * 2
        for _ in range(2, n + 1):
            a, b = b, a + b

        return b


class RecoveryHandler(ABC):
    """Abstract base class for recovery handlers."""

    @abstractmethod
    def can_handle(self, exception: Exception) -> bool:
        """Check if this handler can handle the exception.

        Args:
            exception: The exception to check

        Returns:
            True if can handle
        """
        pass

    @abstractmethod
    def handle(self, exception: Exception, context: dict[str, Any]) -> Any:
        """Handle the exception.

        Args:
            exception: The exception to handle
            context: Context information

        Returns:
            Recovery result or None
        """
        pass


class DefaultRecoveryHandler(RecoveryHandler):
    """Default recovery handler that logs and re-raises."""

    def can_handle(self, exception: Exception) -> bool:
        """Can handle any exception.

        Args:
            exception: The exception

        Returns:
            Always True
        """
        return True

    def handle(self, exception: Exception, context: dict[str, Any]) -> Any:
        """Log and re-raise.

        Args:
            exception: The exception
            context: Context information

        Raises:
            The original exception
        """
        logger.error(f"Unrecoverable error in {context.get('method', 'unknown')}: {exception}")
        raise


class ErrorRecoveryAspect:
    """Provides automatic error recovery.

    Port of ErrorRecoveryAspect from Qontinui framework.

    Features:
    - Configurable retry policies
    - Multiple retry strategies
    - Custom recovery handlers
    - Circuit breaker pattern
    - Error rate tracking
    """

    def __init__(self, enabled: bool = True, default_policy: RetryPolicy | None = None):
        """Initialize the aspect.

        Args:
            enabled: Whether recovery is enabled
            default_policy: Default retry policy
        """
        self.enabled = enabled
        self.default_policy = default_policy or RetryPolicy()

        # Recovery handlers
        self._handlers: list[RecoveryHandler] = [DefaultRecoveryHandler()]

        # Method-specific policies
        self._method_policies: dict[str, RetryPolicy] = {}

        # Error tracking
        self._error_counts: dict[str, int] = {}
        self._recovery_counts: dict[str, int] = {}

        # Circuit breaker state
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

    def with_recovery(
        self, policy: RetryPolicy | None = None, fallback: Callable[..., Any] | None = None
    ) -> Callable[..., Any]:
        """Decorator to add error recovery to a method.

        Args:
            policy: Retry policy to use
            fallback: Fallback function if all retries fail

        Returns:
            Decorator function
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            method_name = f"{func.__module__}.{func.__name__}"

            # Store method-specific policy
            if policy:
                self._method_policies[method_name] = policy

            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)

                # Get applicable policy
                retry_policy = policy or self._method_policies.get(method_name, self.default_policy)

                # Check circuit breaker
                if method_name in self._circuit_breakers:
                    breaker = self._circuit_breakers[method_name]
                    if breaker.is_open():
                        logger.warning(f"Circuit breaker open for {method_name}")
                        if fallback:
                            return fallback()
                        raise RuntimeError(f"Circuit breaker open for {method_name}")

                last_exception = None

                for attempt in range(retry_policy.max_attempts):
                    try:
                        if attempt > 0:
                            # Calculate and apply delay
                            delay_ms = retry_policy.get_delay(attempt - 1)
                            logger.info(
                                f"Retry {attempt}/{retry_policy.max_attempts} for {method_name} "
                                f"after {delay_ms:.0f}ms delay"
                            )
                            time.sleep(delay_ms / 1000)

                        # Execute function
                        result = func(*args, **kwargs)

                        # Success - update metrics
                        if attempt > 0:
                            self._recovery_counts[method_name] = (
                                self._recovery_counts.get(method_name, 0) + 1
                            )
                            logger.info(
                                f"Recovery successful for {method_name} on attempt {attempt + 1}"
                            )

                        # Update circuit breaker
                        if method_name in self._circuit_breakers:
                            self._circuit_breakers[method_name].record_success()

                        return result

                    except Exception as e:
                        last_exception = e

                        # Update error count
                        self._error_counts[method_name] = self._error_counts.get(method_name, 0) + 1

                        # Check if should retry
                        if not retry_policy.should_retry(e):
                            logger.error(f"Non-recoverable error in {method_name}: {e}")
                            raise

                        # Check if out of attempts
                        if attempt >= retry_policy.max_attempts - 1:
                            logger.error(
                                f"All {retry_policy.max_attempts} recovery attempts failed "
                                f"for {method_name}"
                            )

                            # Update circuit breaker
                            if method_name in self._circuit_breakers:
                                self._circuit_breakers[method_name].record_failure()

                            # Try recovery handlers
                            context = {
                                "method": method_name,
                                "attempts": retry_policy.max_attempts,
                                "args": args,
                                "kwargs": kwargs,
                            }

                            for handler in self._handlers:
                                if handler.can_handle(e):
                                    try:
                                        return handler.handle(e, context)
                                    except Exception:
                                        continue

                            # Try fallback
                            if fallback:
                                logger.info(f"Using fallback for {method_name}")
                                try:
                                    return fallback()
                                except Exception as fallback_error:
                                    logger.error(f"Fallback also failed: {fallback_error}")

                            # Re-raise original exception
                            raise

                        logger.warning(
                            f"Recoverable error in {method_name} "
                            f"(attempt {attempt + 1}/{retry_policy.max_attempts}): {e}"
                        )

                # Should not reach here
                if last_exception:
                    raise last_exception

            return wrapper

        return decorator

    def add_handler(self, handler: RecoveryHandler) -> None:
        """Add a custom recovery handler.

        Args:
            handler: Recovery handler to add
        """
        self._handlers.insert(0, handler)  # Insert at beginning for priority

    def enable_circuit_breaker(
        self, method_name: str, failure_threshold: int = 5, reset_timeout_seconds: int = 60
    ) -> None:
        """Enable circuit breaker for a method.

        Args:
            method_name: Method to protect
            failure_threshold: Failures before opening circuit
            reset_timeout_seconds: Time before attempting reset
        """
        self._circuit_breakers[method_name] = CircuitBreaker(
            failure_threshold, reset_timeout_seconds
        )

    def get_error_stats(self) -> dict[str, dict[str, int]]:
        """Get error and recovery statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {}

        for method in set(self._error_counts.keys()) | set(self._recovery_counts.keys()):
            stats[method] = {
                "errors": self._error_counts.get(method, 0),
                "recoveries": self._recovery_counts.get(method, 0),
            }

        return stats


@dataclass
class CircuitBreaker:
    """Simple circuit breaker implementation."""

    failure_threshold: int
    """Number of failures before opening."""

    reset_timeout_seconds: int
    """Seconds before attempting reset."""

    failure_count: int = 0
    """Current failure count."""

    last_failure_time: float | None = None
    """Time of last failure."""

    state: str = "closed"
    """Current state (closed, open, half_open)."""

    def is_open(self) -> bool:
        """Check if circuit is open.

        Returns:
            True if circuit is open
        """
        if self.state == "closed":
            return False

        # Check if should transition to half-open
        if self.state == "open" and self.last_failure_time:
            if time.time() - self.last_failure_time > self.reset_timeout_seconds:
                self.state = "half_open"
                return False

        return self.state == "open"

    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == "half_open":
            self.state = "closed"
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"


# Global instance
_error_recovery_aspect = ErrorRecoveryAspect()


def with_error_recovery(
    policy: RetryPolicy | None = None, fallback: Callable[..., Any] | None = None
) -> Callable[..., Any]:
    """Decorator for error recovery.

    Args:
        policy: Retry policy
        fallback: Fallback function

    Returns:
        Decorator function
    """
    return _error_recovery_aspect.with_recovery(policy, fallback)


def get_error_recovery_aspect() -> ErrorRecoveryAspect:
    """Get the global error recovery aspect.

    Returns:
        The error recovery aspect
    """
    return _error_recovery_aspect
