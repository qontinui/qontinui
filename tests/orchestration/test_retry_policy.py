"""Tests for retry policy."""

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import directly from module to avoid cv2 dependency
from qontinui.orchestration.retry_policy import BackoffStrategy, RetryPolicy


class TestRetryPolicy:
    """Test RetryPolicy class."""

    def test_no_retry_policy(self):
        """Test policy with no retries."""
        policy = RetryPolicy.no_retry()

        assert policy.max_retries == 0
        assert not policy.should_retry(0)
        assert not policy.should_retry(1)

    def test_fixed_delay_calculation(self):
        """Test fixed delay calculation."""
        policy = RetryPolicy.with_fixed_delay(max_retries=3, delay=2.0)

        assert policy.calculate_delay(0) == 2.0
        assert policy.calculate_delay(1) == 2.0
        assert policy.calculate_delay(2) == 2.0
        assert policy.calculate_delay(10) == 2.0

    def test_linear_backoff_calculation(self):
        """Test linear backoff calculation."""
        policy = RetryPolicy.with_linear_backoff(max_retries=3, base_delay=1.0)

        assert policy.calculate_delay(0) == 1.0  # 1.0 * 1
        assert policy.calculate_delay(1) == 2.0  # 1.0 * 2
        assert policy.calculate_delay(2) == 3.0  # 1.0 * 3

    def test_exponential_backoff_calculation(self):
        """Test exponential backoff calculation."""
        policy = RetryPolicy.with_exponential_backoff(max_retries=5, base_delay=1.0)

        assert policy.calculate_delay(0) == 1.0  # 1.0 * 2^0
        assert policy.calculate_delay(1) == 2.0  # 1.0 * 2^1
        assert policy.calculate_delay(2) == 4.0  # 1.0 * 2^2
        assert policy.calculate_delay(3) == 8.0  # 1.0 * 2^3
        assert policy.calculate_delay(4) == 16.0  # 1.0 * 2^4

    def test_max_delay_cap(self):
        """Test that max_delay caps the calculated delay."""
        policy = RetryPolicy.with_exponential_backoff(
            max_retries=10, base_delay=1.0, max_delay=10.0
        )

        # Should cap at 10.0
        assert policy.calculate_delay(10) == 10.0
        assert policy.calculate_delay(20) == 10.0

    def test_should_retry_within_limit(self):
        """Test retry decision within retry limit."""
        policy = RetryPolicy(max_retries=3)

        assert policy.should_retry(0)
        assert policy.should_retry(1)
        assert policy.should_retry(2)
        assert not policy.should_retry(3)
        assert not policy.should_retry(4)

    def test_should_retry_with_condition(self):
        """Test retry decision with retry condition."""

        def retry_on_timeout(error: Exception) -> bool:
            return isinstance(error, TimeoutError)

        policy = RetryPolicy(max_retries=3, retry_condition=retry_on_timeout)

        # Should retry for TimeoutError
        assert policy.should_retry(0, TimeoutError())

        # Should not retry for other errors
        assert not policy.should_retry(0, ValueError())
        assert not policy.should_retry(0, RuntimeError())

    def test_should_retry_no_condition_with_error(self):
        """Test retry decision with error but no condition."""
        policy = RetryPolicy(max_retries=3)

        # Should retry regardless of error type when no condition
        assert policy.should_retry(0, ValueError())
        assert policy.should_retry(0, TimeoutError())

    @patch("time.sleep")
    def test_wait_for_retry_fixed(self, mock_sleep):
        """Test waiting for retry with fixed delay."""
        policy = RetryPolicy.with_fixed_delay(max_retries=3, delay=1.5)

        policy.wait_for_retry(0)
        mock_sleep.assert_called_once_with(1.5)

        mock_sleep.reset_mock()
        policy.wait_for_retry(1)
        mock_sleep.assert_called_once_with(1.5)

    @patch("time.sleep")
    def test_wait_for_retry_exponential(self, mock_sleep):
        """Test waiting for retry with exponential backoff."""
        policy = RetryPolicy.with_exponential_backoff(max_retries=3, base_delay=1.0)

        policy.wait_for_retry(0)
        mock_sleep.assert_called_with(1.0)

        mock_sleep.reset_mock()
        policy.wait_for_retry(1)
        mock_sleep.assert_called_with(2.0)

        mock_sleep.reset_mock()
        policy.wait_for_retry(2)
        mock_sleep.assert_called_with(4.0)

    @patch("time.sleep")
    def test_wait_for_retry_zero_delay(self, mock_sleep):
        """Test that zero delay doesn't sleep."""
        policy = RetryPolicy(max_retries=3, base_delay=0.0)

        policy.wait_for_retry(0)
        mock_sleep.assert_not_called()

    def test_continue_on_error_flag(self):
        """Test continue_on_error flag."""
        policy_continue = RetryPolicy(max_retries=0, continue_on_error=True)
        policy_stop = RetryPolicy(max_retries=0, continue_on_error=False)

        assert policy_continue.continue_on_error
        assert not policy_stop.continue_on_error

    def test_backoff_strategy_enum(self):
        """Test BackoffStrategy enum values."""
        assert BackoffStrategy.FIXED
        assert BackoffStrategy.LINEAR
        assert BackoffStrategy.EXPONENTIAL

    def test_custom_retry_policy(self):
        """Test creating custom retry policy."""
        policy = RetryPolicy(
            max_retries=5,
            base_delay=0.5,
            backoff_strategy=BackoffStrategy.LINEAR,
            max_delay=10.0,
            continue_on_error=True,
        )

        assert policy.max_retries == 5
        assert policy.base_delay == 0.5
        assert policy.backoff_strategy == BackoffStrategy.LINEAR
        assert policy.max_delay == 10.0
        assert policy.continue_on_error

    def test_actual_delay_timing(self):
        """Test actual delay timing (integration test)."""
        policy = RetryPolicy.with_fixed_delay(max_retries=1, delay=0.1)

        start = time.time()
        policy.wait_for_retry(0)
        duration = time.time() - start

        # Should be approximately 0.1 seconds (with some tolerance)
        assert 0.08 < duration < 0.15

    def test_retry_condition_called_correctly(self):
        """Test that retry condition is called with correct error."""
        called_with = []

        def track_calls(error: Exception) -> bool:
            called_with.append(error)
            return True

        policy = RetryPolicy(max_retries=3, retry_condition=track_calls)

        test_error = ValueError("test error")
        policy.should_retry(0, test_error)

        assert len(called_with) == 1
        assert called_with[0] is test_error


class TestRetryPolicyFactoryMethods:
    """Test RetryPolicy factory methods."""

    def test_no_retry_factory(self):
        """Test no_retry factory method."""
        policy = RetryPolicy.no_retry()

        assert policy.max_retries == 0
        assert policy.backoff_strategy == BackoffStrategy.FIXED

    def test_with_fixed_delay_factory(self):
        """Test with_fixed_delay factory method."""
        policy = RetryPolicy.with_fixed_delay(max_retries=3, delay=2.5)

        assert policy.max_retries == 3
        assert policy.base_delay == 2.5
        assert policy.backoff_strategy == BackoffStrategy.FIXED

    def test_with_exponential_backoff_factory(self):
        """Test with_exponential_backoff factory method."""
        policy = RetryPolicy.with_exponential_backoff(
            max_retries=5, base_delay=1.0, max_delay=20.0
        )

        assert policy.max_retries == 5
        assert policy.base_delay == 1.0
        assert policy.max_delay == 20.0
        assert policy.backoff_strategy == BackoffStrategy.EXPONENTIAL

    def test_with_linear_backoff_factory(self):
        """Test with_linear_backoff factory method."""
        policy = RetryPolicy.with_linear_backoff(max_retries=4, base_delay=0.5, max_delay=15.0)

        assert policy.max_retries == 4
        assert policy.base_delay == 0.5
        assert policy.max_delay == 15.0
        assert policy.backoff_strategy == BackoffStrategy.LINEAR
