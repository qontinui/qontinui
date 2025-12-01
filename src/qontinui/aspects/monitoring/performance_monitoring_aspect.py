"""Performance monitoring aspect - ported from Qontinui framework.

Provides comprehensive performance monitoring for all operations.
"""

import logging
import statistics
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from threading import Lock
from typing import Any, cast

logger = logging.getLogger(__name__)


@dataclass
class MethodPerformanceStats:
    """Performance statistics for a method."""

    method_name: str
    """Name of the method."""

    total_calls: int = 0
    """Total number of calls."""

    total_time_ms: float = 0.0
    """Total execution time in milliseconds."""

    min_time_ms: float = float("inf")
    """Minimum execution time."""

    max_time_ms: float = 0.0
    """Maximum execution time."""

    recent_times: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    """Recent execution times for percentile calculation."""

    last_call_time: datetime | None = None
    """Timestamp of last call."""

    error_count: int = 0
    """Number of errors."""

    @property
    def average_time_ms(self) -> float:
        """Calculate average execution time."""
        if self.total_calls == 0:
            return 0.0
        return self.total_time_ms / self.total_calls

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 100.0
        return ((self.total_calls - self.error_count) / self.total_calls) * 100

    def get_percentile(self, percentile: int) -> float:
        """Get percentile of recent execution times.

        Args:
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value or 0 if no data
        """
        if not self.recent_times:
            return 0.0

        sorted_times = sorted(self.recent_times)
        index = int(len(sorted_times) * percentile / 100)
        return sorted_times[min(index, len(sorted_times) - 1)]

    def update(self, duration_ms: float, success: bool = True) -> None:
        """Update statistics with new execution.

        Args:
            duration_ms: Execution duration in milliseconds
            success: Whether execution succeeded
        """
        self.total_calls += 1
        self.total_time_ms += duration_ms
        self.min_time_ms = min(self.min_time_ms, duration_ms)
        self.max_time_ms = max(self.max_time_ms, duration_ms)
        self.recent_times.append(duration_ms)
        self.last_call_time = datetime.now()

        if not success:
            self.error_count += 1


class PerformanceMonitoringAspect:
    """Provides comprehensive performance monitoring.

    Port of PerformanceMonitoringAspect from Qontinui framework.

    Features:
    - Method-level execution time tracking
    - Statistical analysis (min, max, avg, percentiles)
    - Performance trend detection
    - Slow operation alerts
    - Periodic performance reports
    - Memory usage correlation
    """

    def __init__(
        self,
        enabled: bool = True,
        alert_threshold_ms: float = 10000,
        report_interval_seconds: int = 300,
        track_memory: bool = True,
        warning_threshold_ms: float = 5000,
    ) -> None:
        """Initialize the aspect.

        Args:
            enabled: Whether monitoring is enabled
            alert_threshold_ms: Threshold for slow operation alerts
            report_interval_seconds: Interval for periodic reports
            track_memory: Whether to track memory usage
            warning_threshold_ms: Threshold for performance warnings
        """
        self.enabled = enabled
        self.alert_threshold_ms = alert_threshold_ms
        self.report_interval_seconds = report_interval_seconds
        self.track_memory = track_memory
        self.warning_threshold_ms = warning_threshold_ms

        # Performance data
        self._stats: dict[str, MethodPerformanceStats] = {}
        self._stats_lock = Lock()

        # Report tracking
        self._last_report_time = datetime.now()

        # Trend detection
        self._performance_trends: dict[str, deque[float]] = {}

    def monitor(
        self, name: str | None = None, track_args: bool = False
    ) -> Callable[..., Any]:
        """Decorator to monitor method performance.

        Args:
            name: Custom name for the operation
            track_args: Whether to track arguments

        Returns:
            Decorator function
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            operation_name = name or f"{func.__module__}.{func.__name__}"

            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)

                # Start timing
                start_time = time.time()
                start_memory = self._get_memory_usage() if self.track_memory else None

                success = False
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    success = True
                    return result

                except Exception as e:
                    # Log error but re-raise
                    logger.error(f"Error in {operation_name}: {e}")
                    raise

                finally:
                    # Calculate metrics
                    duration_ms = (time.time() - start_time) * 1000

                    # Update statistics
                    self._update_stats(operation_name, duration_ms, success)

                    # Check for slow operation
                    self._check_performance_threshold(operation_name, duration_ms)

                    # Track memory if enabled
                    if self.track_memory and start_memory is not None:
                        end_memory = self._get_memory_usage()
                        memory_delta = end_memory - start_memory
                        if memory_delta > 100:  # More than 100MB
                            logger.warning(
                                f"{operation_name} used {memory_delta:.2f}MB of memory"
                            )

                    # Check if report is due
                    self._check_report_interval()

            return wrapper

        return decorator

    def _update_stats(
        self, method_name: str, duration_ms: float, success: bool
    ) -> None:
        """Update performance statistics.

        Args:
            method_name: Name of the method
            duration_ms: Execution duration
            success: Whether execution succeeded
        """
        with self._stats_lock:
            if method_name not in self._stats:
                self._stats[method_name] = MethodPerformanceStats(method_name)

            self._stats[method_name].update(duration_ms, success)

            # Update trend tracking
            if method_name not in self._performance_trends:
                self._performance_trends[method_name] = deque(maxlen=100)
            self._performance_trends[method_name].append(duration_ms)

    def _check_performance_threshold(
        self, method_name: str, duration_ms: float
    ) -> None:
        """Check if operation exceeded performance thresholds.

        Args:
            method_name: Name of the method
            duration_ms: Execution duration
        """
        if duration_ms > self.alert_threshold_ms:
            logger.error(
                f"PERFORMANCE ALERT: {method_name} took {duration_ms:.2f}ms "
                f"(threshold: {self.alert_threshold_ms}ms)"
            )
        elif duration_ms > self.warning_threshold_ms:
            logger.warning(
                f"Performance warning: {method_name} took {duration_ms:.2f}ms"
            )

    def _check_report_interval(self) -> None:
        """Check if it's time to generate a performance report."""
        now = datetime.now()
        if (now - self._last_report_time).seconds >= self.report_interval_seconds:
            self.generate_report()
            self._last_report_time = now

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        try:
            import psutil

            process = psutil.Process()
            return cast(float, process.memory_info().rss / 1024 / 1024)
        except ImportError:
            return 0.0

    def get_performance_stats(self) -> dict[str, MethodPerformanceStats]:
        """Get all performance statistics.

        Returns:
            Dictionary of method names to stats
        """
        with self._stats_lock:
            return dict(self._stats)

    def get_slowest_operations(self, limit: int = 10) -> list[tuple[Any, ...]]:
        """Get the slowest operations.

        Args:
            limit: Maximum number of operations to return

        Returns:
            List of (method_name, average_time_ms) tuples
        """
        with self._stats_lock:
            operations = [
                (name, stats.average_time_ms) for name, stats in self._stats.items()
            ]
            operations.sort(key=lambda x: x[1], reverse=True)
            return operations[:limit]

    def detect_performance_degradation(
        self, method_name: str, threshold_percent: float = 20
    ) -> bool:
        """Detect if a method's performance has degraded.

        Args:
            method_name: Method to check
            threshold_percent: Percentage increase to consider degradation

        Returns:
            True if performance has degraded
        """
        if method_name not in self._performance_trends:
            return False

        trends = list(self._performance_trends[method_name])
        if len(trends) < 20:
            return False

        # Compare recent average to historical average
        recent = statistics.mean(trends[-10:])
        historical = statistics.mean(trends[:-10])

        if historical == 0:
            return False

        increase_percent = ((recent - historical) / historical) * 100
        return increase_percent > threshold_percent

    def generate_report(self) -> str:
        """Generate a performance report.

        Returns:
            Performance report as string
        """
        report_lines = [
            "=" * 60,
            "PERFORMANCE REPORT",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 60,
            "",
        ]

        with self._stats_lock:
            # Summary
            total_calls = sum(s.total_calls for s in self._stats.values())
            total_time = sum(s.total_time_ms for s in self._stats.values())
            total_errors = sum(s.error_count for s in self._stats.values())

            report_lines.extend(
                [
                    f"Total Operations: {total_calls}",
                    f"Total Time: {total_time:.2f}ms",
                    f"Total Errors: {total_errors}",
                    (
                        f"Overall Success Rate: {((total_calls - total_errors) / total_calls * 100):.2f}%"
                        if total_calls > 0
                        else "N/A"
                    ),
                    "",
                    "TOP 10 SLOWEST OPERATIONS:",
                    "-" * 40,
                ]
            )

            # Slowest operations
            for method, avg_time in self.get_slowest_operations():
                stats = self._stats[method]
                report_lines.append(
                    f"{method}: avg={avg_time:.2f}ms, "
                    f"calls={stats.total_calls}, "
                    f"p95={stats.get_percentile(95):.2f}ms"
                )

            # Performance degradation detection
            report_lines.extend(["", "PERFORMANCE DEGRADATION DETECTED:", "-" * 40])

            degraded = []
            for method_name in self._stats.keys():
                if self.detect_performance_degradation(method_name):
                    degraded.append(method_name)

            if degraded:
                for method in degraded:
                    report_lines.append(f"- {method}")
            else:
                report_lines.append("None detected")

        report = "\\n".join(report_lines)
        logger.info(report)
        return report

    def reset_stats(self) -> None:
        """Reset all performance statistics."""
        with self._stats_lock:
            self._stats.clear()
            self._performance_trends.clear()
        logger.info("Performance statistics reset")


# Global instance
_performance_aspect = PerformanceMonitoringAspect()


def performance_monitored(name: str | None = None) -> Callable[..., Any]:
    """Decorator for performance monitoring.

    Args:
        name: Custom operation name

    Returns:
        Decorator function
    """
    return _performance_aspect.monitor(name)


def get_performance_aspect() -> PerformanceMonitoringAspect:
    """Get the global performance monitoring aspect.

    Returns:
        The performance aspect
    """
    return _performance_aspect
