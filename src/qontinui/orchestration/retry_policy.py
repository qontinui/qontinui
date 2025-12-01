"""Retry policy for action execution.

Provides retry configuration, backoff strategies, and continue-on-error logic.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto


class BackoffStrategy(Enum):
    """Strategy for calculating retry delays."""

    FIXED = auto()
    """Fixed delay between retries."""

    LINEAR = auto()
    """Linearly increasing delay (attempt * base_delay)."""

    EXPONENTIAL = auto()
    """Exponentially increasing delay (base_delay * 2^attempt)."""


@dataclass
class RetryPolicy:
    """Configuration for retry behavior during action execution.

    Provides control over retry attempts, delays, backoff strategies,
    and continue-on-error behavior.
    """

    max_retries: int = 0
    """Maximum number of retry attempts (0 = no retries)."""

    base_delay: float = 1.0
    """Base delay in seconds between retries."""

    backoff_strategy: BackoffStrategy = BackoffStrategy.FIXED
    """Strategy for calculating delay between retries."""

    max_delay: float = 30.0
    """Maximum delay in seconds (caps exponential backoff)."""

    continue_on_error: bool = False
    """If True, continue workflow execution even if action fails."""

    retry_condition: Callable[[Exception], bool] | None = None
    """Optional predicate to determine if an exception should trigger retry."""

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given retry attempt.

        Args:
            attempt: Retry attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        if self.backoff_strategy == BackoffStrategy.FIXED:
            delay = self.base_delay
        elif self.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)
        elif self.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.base_delay * (2**attempt)
        else:
            delay = self.base_delay

        return min(delay, self.max_delay)

    def should_retry(self, attempt: int, error: Exception | None = None) -> bool:
        """Determine if retry should be attempted.

        Args:
            attempt: Current retry attempt number (0-indexed)
            error: Exception that occurred, if any

        Returns:
            True if retry should be attempted
        """
        if attempt >= self.max_retries:
            return False

        if error and self.retry_condition:
            return self.retry_condition(error)

        return True

    def wait_for_retry(self, attempt: int) -> None:
        """Wait for the appropriate delay before retry.

        Args:
            attempt: Current retry attempt number (0-indexed)
        """
        delay = self.calculate_delay(attempt)
        if delay > 0:
            time.sleep(delay)

    @classmethod
    def no_retry(cls) -> "RetryPolicy":
        """Create a policy with no retries.

        Returns:
            RetryPolicy with max_retries=0
        """
        return cls(max_retries=0)

    @classmethod
    def with_fixed_delay(cls, max_retries: int, delay: float) -> "RetryPolicy":
        """Create a policy with fixed delay between retries.

        Args:
            max_retries: Maximum number of retry attempts
            delay: Fixed delay in seconds

        Returns:
            RetryPolicy with fixed backoff
        """
        return cls(
            max_retries=max_retries,
            base_delay=delay,
            backoff_strategy=BackoffStrategy.FIXED,
        )

    @classmethod
    def with_exponential_backoff(
        cls, max_retries: int, base_delay: float = 1.0, max_delay: float = 30.0
    ) -> "RetryPolicy":
        """Create a policy with exponential backoff.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay cap in seconds

        Returns:
            RetryPolicy with exponential backoff
        """
        return cls(
            max_retries=max_retries,
            base_delay=base_delay,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            max_delay=max_delay,
        )

    @classmethod
    def with_linear_backoff(
        cls, max_retries: int, base_delay: float = 1.0, max_delay: float = 30.0
    ) -> "RetryPolicy":
        """Create a policy with linear backoff.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay cap in seconds

        Returns:
            RetryPolicy with linear backoff
        """
        return cls(
            max_retries=max_retries,
            base_delay=base_delay,
            backoff_strategy=BackoffStrategy.LINEAR,
            max_delay=max_delay,
        )
