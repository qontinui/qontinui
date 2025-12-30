"""TimeWrapper - Routes time operations to mock or real implementations (Brobot pattern).

This wrapper provides the routing layer for time operations,
delegating to either MockTime (virtual clock) or real time
based on ExecutionMode.

Architecture:
    Wait/Delay operations (high-level)
      ↓
    TimeWrapper (this layer) ← Routes based on ExecutionMode
      ↓
    ├─ if mock → MockTime → Virtual clock (instant or controlled time)
    └─ if real → time.sleep() → Real system time

This is especially important for deterministic testing where you want
wait operations to complete instantly or at a controlled rate.
"""

import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

from qontinui_schemas.common import utc_now

from .base import BaseWrapper

logger = logging.getLogger(__name__)


class TimeWrapper(BaseWrapper):
    """Wrapper for time operations.

    Routes time operations to either mock or real implementations based on
    ExecutionMode. This follows the Brobot pattern where high-level code is
    agnostic to whether it's running in mock or real mode.

    In mock mode, wait operations can complete instantly (making tests 100x faster)
    or use a virtual clock that can be controlled programmatically.

    Example:
        # Initialize wrapper
        wrapper = TimeWrapper()

        # Wait 5 seconds (automatically routed to mock or real)
        wrapper.wait(5.0)

        # In mock mode: Returns instantly (or virtual 5 seconds)
        # In real mode: Actually waits 5 seconds

    Attributes:
        mock_time: MockTime instance for virtual clock
    """

    def __init__(self) -> None:
        """Initialize TimeWrapper.

        Sets up both mock and real implementations. The actual implementation
        used is determined at runtime based on ExecutionMode.
        """
        super().__init__()

        # Lazy initialization
        self._mock_time = None

        logger.debug("TimeWrapper initialized")

    @property
    def mock_time(self):
        """Get MockTime instance (lazy initialization).

        Returns:
            MockTime instance
        """
        if self._mock_time is None:
            from ..mock.mock_time import MockTime

            self._mock_time = MockTime()
            logger.debug("MockTime initialized")
        return self._mock_time

    def wait(self, seconds: float) -> None:
        """Wait for specified duration.

        Args:
            seconds: Duration to wait in seconds

        Example:
            wrapper = TimeWrapper()
            wrapper.wait(2.5)  # Waits 2.5 seconds (or instant in mock mode)
        """
        if self.is_mock_mode():
            logger.debug(f"TimeWrapper.wait (MOCK): {seconds}s")
            self.mock_time.wait(seconds)
        else:
            logger.debug(f"TimeWrapper.wait (REAL): {seconds}s")
            time.sleep(seconds)

    def now(self) -> datetime:
        """Get current time.

        Returns:
            Current datetime

        Example:
            wrapper = TimeWrapper()
            current_time = wrapper.now()

        Note:
            In mock mode, returns virtual time from MockTime clock.
            In real mode, returns actual system time.
        """
        if self.is_mock_mode():
            logger.debug("TimeWrapper.now (MOCK)")
            return cast(datetime, self.mock_time.now())
        else:
            logger.debug("TimeWrapper.now (REAL)")
            result: datetime = utc_now()
            return result

    def wait_until(
        self,
        condition: Callable[[], bool],
        timeout: float = 10.0,
        poll_interval: float = 0.5,
    ) -> bool:
        """Wait until condition becomes true.

        Args:
            condition: Function that returns True when condition is met
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check condition

        Returns:
            True if condition met, False if timeout

        Example:
            def is_ready():
                return check_if_app_loaded()

            wrapper = TimeWrapper()
            success = wrapper.wait_until(is_ready, timeout=30.0)
        """
        if self.is_mock_mode():
            logger.debug(f"TimeWrapper.wait_until (MOCK): timeout={timeout}s")
            return cast(bool, self.mock_time.wait_until(condition, timeout, poll_interval))
        else:
            logger.debug(f"TimeWrapper.wait_until (REAL): timeout={timeout}s")
            return self._wait_until_real(condition, timeout, poll_interval)

    def _wait_until_real(
        self,
        condition: Callable[[], bool],
        timeout: float,
        poll_interval: float,
    ) -> bool:
        """Real implementation of wait_until.

        Args:
            condition: Condition function
            timeout: Maximum wait time
            poll_interval: Poll interval

        Returns:
            True if condition met, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if condition():
                return True
            time.sleep(poll_interval)

        return False

    def measure(self, func: Callable[[], Any]) -> tuple[Any, float]:
        """Measure execution time of a function.

        Args:
            func: Function to measure

        Returns:
            Tuple of (result, duration_seconds)

        Example:
            wrapper = TimeWrapper()
            result, duration = wrapper.measure(lambda: expensive_operation())
            print(f"Operation took {duration:.3f} seconds")
        """
        if self.is_mock_mode():
            logger.debug("TimeWrapper.measure (MOCK)")
            return cast(tuple[Any, float], self.mock_time.measure(func))
        else:
            logger.debug("TimeWrapper.measure (REAL)")
            start = time.time()
            result = func()
            duration = time.time() - start
            return result, duration

    def timestamp(self) -> float:
        """Get current timestamp.

        Returns:
            Current timestamp (seconds since epoch)

        Example:
            wrapper = TimeWrapper()
            ts = wrapper.timestamp()
        """
        if self.is_mock_mode():
            logger.debug("TimeWrapper.timestamp (MOCK)")
            return cast(float, self.mock_time.timestamp())
        else:
            logger.debug("TimeWrapper.timestamp (REAL)")
            return time.time()
