"""MockTime - Simulates time operations with virtual clock (Brobot pattern).

Provides mock time operations for deterministic testing. Uses a virtual clock
that can advance instantly or at a controlled rate, making tests 10-100x faster
and completely deterministic.

This enables:
- Instant time progression (no real waiting)
- Deterministic testing (same results every time)
- Fast test execution (waits complete instantly)
- Virtual clock control (manual time advancement)
"""

import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from qontinui_schemas.common import utc_now

logger = logging.getLogger(__name__)


class MockTime:
    """Mock time implementation with virtual clock.

    Simulates time operations using a virtual clock that can be controlled
    programmatically. Wait operations complete instantly by default, but
    the virtual clock can be configured to advance at a different rate.

    Example:
        time = MockTime()
        time.wait(5.0)  # Returns instantly, virtual clock advances 5 seconds

        # Check virtual time
        start = time.now()
        time.wait(10.0)
        end = time.now()
        assert (end - start).seconds == 10  # Virtual time advanced

    Attributes:
        instant_mode: If True, waits complete instantly (default)
        time_scale: Virtual time scale (1.0 = real time, 0.0 = instant)
        virtual_time: Current virtual time
    """

    def __init__(self, instant_mode: bool = True, time_scale: float = 0.0) -> None:
        """Initialize MockTime.

        Args:
            instant_mode: If True, waits complete instantly
            time_scale: Virtual time scale (0.0 = instant, 1.0 = real time)
        """
        self.instant_mode = instant_mode
        self.time_scale = time_scale

        # Virtual clock starts at initialization time
        self.virtual_time: datetime = utc_now()
        self.start_time: datetime = self.virtual_time

        logger.debug(f"MockTime initialized (instant_mode={instant_mode}, time_scale={time_scale})")

    def wait(self, seconds: float) -> None:
        """Wait for specified duration (mock).

        In instant mode, returns immediately but advances virtual clock.
        Otherwise, advances virtual clock and may sleep based on time_scale.

        Args:
            seconds: Duration to wait

        Example:
            time = MockTime()
            time.wait(5.0)  # Returns instantly
        """
        logger.debug(f"MockTime.wait: {seconds}s (instant_mode={self.instant_mode})")

        # Advance virtual clock
        self.virtual_time += timedelta(seconds=seconds)

        if not self.instant_mode and self.time_scale > 0:
            # Sleep for scaled time
            import time as real_time

            scaled_duration = seconds * self.time_scale
            real_time.sleep(scaled_duration)

    def now(self) -> datetime:
        """Get current virtual time.

        Returns:
            Current virtual datetime

        Example:
            time = MockTime()
            current = time.now()
        """
        logger.debug(f"MockTime.now: {self.virtual_time}")
        result: datetime = self.virtual_time
        return result

    def timestamp(self) -> float:
        """Get current virtual timestamp.

        Returns:
            Virtual timestamp (seconds since epoch)

        Example:
            time = MockTime()
            ts = time.timestamp()
        """
        result: float = self.virtual_time.timestamp()
        return result

    def wait_until(
        self,
        condition: Callable[[], bool],
        timeout: float = 10.0,
        poll_interval: float = 0.5,
    ) -> bool:
        """Wait until condition becomes true (mock).

        In mock mode, checks condition immediately. If not met, advances
        virtual clock by timeout and checks again. This makes conditional
        waits very fast while still tracking virtual time.

        Args:
            condition: Function that returns True when condition is met
            timeout: Maximum time to wait
            poll_interval: How often to check (affects virtual time advancement)

        Returns:
            True if condition met, False if timeout

        Example:
            def is_ready():
                return app.is_loaded()

            time = MockTime()
            success = time.wait_until(is_ready, timeout=30.0)
        """
        logger.debug(f"MockTime.wait_until: timeout={timeout}s")

        start_virtual_time = self.virtual_time
        max_checks = int(timeout / poll_interval) if poll_interval > 0 else 1

        for _i in range(max_checks):
            if condition():
                elapsed = self.virtual_time - start_virtual_time
                logger.debug(f"Condition met after virtual {elapsed.total_seconds()}s")
                return True

            # Advance virtual clock
            self.virtual_time += timedelta(seconds=poll_interval)

        # Timeout
        logger.debug(f"Condition not met after virtual {timeout}s")
        return False

    def measure(self, func: Callable[[], Any]) -> tuple[Any, float]:
        """Measure execution time (virtual).

        Measures virtual time elapsed during function execution.

        Args:
            func: Function to measure

        Returns:
            Tuple of (result, virtual_duration_seconds)

        Example:
            time = MockTime()
            result, duration = time.measure(lambda: expensive_operation())
        """
        start = self.virtual_time
        result = func()
        end = self.virtual_time
        duration = (end - start).total_seconds()

        logger.debug(f"MockTime.measure: virtual {duration}s")
        return result, duration

    # Virtual clock control methods

    def set_time(self, new_time: datetime) -> None:
        """Set virtual clock to specific time.

        Args:
            new_time: New virtual time

        Example:
            time = MockTime()
            time.set_time(datetime(2025, 1, 1, 12, 0, 0))
        """
        self.virtual_time = new_time
        logger.debug(f"Virtual time set to: {new_time}")

    def advance(self, seconds: float) -> None:
        """Advance virtual clock by specified duration.

        Args:
            seconds: Seconds to advance

        Example:
            time = MockTime()
            time.advance(3600)  # Advance 1 hour
        """
        self.virtual_time += timedelta(seconds=seconds)
        logger.debug(f"Virtual time advanced by {seconds}s to {self.virtual_time}")

    def reset(self) -> None:
        """Reset virtual clock to start time.

        Example:
            time = MockTime()
            time.wait(100)
            time.reset()  # Back to start time
        """
        self.virtual_time = self.start_time
        logger.debug("Virtual time reset to start time")

    def get_elapsed_time(self) -> timedelta:
        """Get elapsed virtual time since initialization.

        Returns:
            Elapsed virtual time

        Example:
            time = MockTime()
            time.wait(10)
            elapsed = time.get_elapsed_time()
            assert elapsed.seconds == 10
        """
        result: timedelta = self.virtual_time - self.start_time
        return result

    def set_instant_mode(self, instant: bool) -> None:
        """Set instant mode.

        Args:
            instant: If True, waits complete instantly

        Example:
            time = MockTime(instant_mode=False)
            time.set_instant_mode(True)  # Switch to instant
        """
        self.instant_mode = instant
        logger.debug(f"Instant mode set to: {instant}")

    def set_time_scale(self, scale: float) -> None:
        """Set virtual time scale.

        Args:
            scale: Time scale (0.0 = instant, 1.0 = real time, 2.0 = 2x speed)

        Example:
            time = MockTime()
            time.set_time_scale(0.1)  # 10x faster than real time
        """
        self.time_scale = scale
        logger.debug(f"Time scale set to: {scale}")
