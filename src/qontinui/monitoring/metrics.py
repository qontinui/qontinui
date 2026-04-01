"""Metrics collection and monitoring for Qontinui using Prometheus.

This replaces Brobot's monitor/diagnostics packages with industry-standard
Prometheus metrics and health checks.
"""

import time
import traceback
from collections.abc import Callable
from threading import Thread
from typing import Any, TypedDict, cast

import psutil
from prometheus_client import REGISTRY, Counter, Gauge, Histogram, Summary, generate_latest
from prometheus_client import start_http_server as prometheus_start_server
from qontinui_schemas.common import utc_now

from ..config import get_settings
from ..logging import get_logger

logger = get_logger(__name__)


class HealthCheckInfo(TypedDict):
    """Type definition for health check information."""

    func: Callable[[], bool]
    critical: bool


# Define Prometheus metrics

# Action metrics
action_counter = Counter(
    "qontinui_actions_total",
    "Total number of actions executed",
    ["action_type", "status"],
)

action_duration = Histogram(
    "qontinui_action_duration_seconds",
    "Action execution duration in seconds",
    ["action_type"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

action_retry_counter = Counter(
    "qontinui_action_retries_total", "Total number of action retries", ["action_type"]
)

# State metrics
state_transitions = Counter(
    "qontinui_state_transitions_total",
    "Total number of state transitions",
    ["from_state", "to_state", "success"],
)

state_activation_duration = Histogram(
    "qontinui_state_activation_seconds",
    "State activation duration in seconds",
    ["state"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)

active_states = Gauge("qontinui_active_states", "Number of currently active states")

# Matching metrics
match_attempts = Counter(
    "qontinui_match_attempts_total",
    "Total number of matching attempts",
    ["match_type", "success"],
)

match_accuracy = Histogram(
    "qontinui_match_accuracy",
    "Matching accuracy scores",
    ["match_type"],
    buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0),
)

match_duration = Histogram(
    "qontinui_match_duration_seconds",
    "Matching operation duration",
    ["match_type"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# System metrics
cpu_usage = Gauge("qontinui_cpu_usage_percent", "CPU usage percentage")

memory_usage = Gauge("qontinui_memory_usage_bytes", "Memory usage in bytes")

thread_count = Gauge("qontinui_thread_count", "Number of active threads")

# Storage metrics
storage_operations = Counter(
    "qontinui_storage_operations_total",
    "Total storage operations",
    ["operation", "storage_type", "status"],
)

storage_size = Gauge("qontinui_storage_size_bytes", "Storage size in bytes", ["storage_type"])

# Error metrics
error_counter = Counter(
    "qontinui_errors_total", "Total number of errors", ["error_type", "component"]
)

# Performance metrics
operation_latency = Summary(
    "qontinui_operation_latency_seconds", "Operation latency", ["operation"]
)


class MetricsCollector:
    """Collect and expose metrics for monitoring.

    Features:
        - Automatic system metrics collection
        - Custom metric registration
        - HTTP endpoint for Prometheus scraping
        - Health check integration
    """

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self.settings = get_settings()
        self.start_time: float = time.time()
        self._running = False
        self._thread: Thread | None = None
        self._custom_metrics: dict[str, Any] = {}

    def start(self, port: int | None = None) -> None:
        """Start metrics collection and HTTP server.

        Args:
            port: Port for metrics endpoint (defaults to settings)
        """
        if self._running:
            logger.warning("Metrics collector already running")
            return

        port = port or self.settings.metrics_port  # type: ignore[attr-defined]

        # Start Prometheus HTTP server
        try:
            prometheus_start_server(port)
            logger.info(
                "metrics_server_started",
                port=port,
                endpoint=f"http://localhost:{port}/metrics",
            )
        except Exception as e:
            logger.error("metrics_server_failed", error=str(e))
            raise

        # Start system metrics collection thread
        self._running = True
        self._thread = Thread(target=self._collect_system_metrics, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop metrics collection."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _collect_system_metrics(self) -> None:
        """Continuously collect system metrics."""
        while self._running:
            try:
                # CPU usage
                cpu_usage.set(psutil.cpu_percent(interval=1))

                # Memory usage
                process = psutil.Process()
                memory_usage.set(process.memory_info().rss)

                # Thread count
                thread_count.set(process.num_threads())

                # Wait before next collection
                time.sleep(self.settings.health_check_interval)  # type: ignore[attr-defined]

            except Exception as e:
                logger.error("system_metrics_collection_failed", error=str(e))
                time.sleep(5)

    # Action metrics methods

    def record_action(
        self, action_type: str, duration: float, success: bool, retries: int = 0
    ) -> None:
        """Record action metrics.

        Args:
            action_type: Type of action
            duration: Execution duration in seconds
            success: Whether action succeeded
            retries: Number of retries
        """
        status = "success" if success else "failure"
        action_counter.labels(action_type=action_type, status=status).inc()
        action_duration.labels(action_type=action_type).observe(duration)

        if retries > 0:
            action_retry_counter.labels(action_type=action_type).inc(retries)

        logger.debug(
            "action_metrics_recorded",
            action_type=action_type,
            duration=duration,
            success=success,
            retries=retries,
        )

    # State metrics methods

    def record_transition(
        self,
        from_state: str,
        to_state: str,
        success: bool = True,
        duration: float | None = None,
    ) -> None:
        """Record state transition.

        Args:
            from_state: Source state
            to_state: Target state
            success: Whether transition succeeded
            duration: Optional transition duration
        """
        state_transitions.labels(
            from_state=from_state, to_state=to_state, success=str(success)
        ).inc()

        if duration is not None:
            state_activation_duration.labels(state=to_state).observe(duration)

    def set_active_states(self, count: int) -> None:
        """Set number of active states.

        Args:
            count: Number of active states
        """
        active_states.set(count)

    # Matching metrics methods

    def record_match(
        self,
        match_type: str,
        success: bool,
        accuracy: float | None = None,
        duration: float | None = None,
    ) -> None:
        """Record matching attempt.

        Args:
            match_type: Type of matching (deterministic, semantic, hybrid)
            success: Whether match was found
            accuracy: Optional accuracy score
            duration: Optional match duration
        """
        match_attempts.labels(match_type=match_type, success=str(success)).inc()

        if accuracy is not None:
            match_accuracy.labels(match_type=match_type).observe(accuracy)

        if duration is not None:
            match_duration.labels(match_type=match_type).observe(duration)

    # Storage metrics methods

    def record_storage_operation(
        self, operation: str, storage_type: str, success: bool, size: int | None = None
    ) -> None:
        """Record storage operation.

        Args:
            operation: Operation type (read, write, delete)
            storage_type: Storage type (json, pickle, database)
            success: Whether operation succeeded
            size: Optional data size in bytes
        """
        status = "success" if success else "failure"
        storage_operations.labels(
            operation=operation, storage_type=storage_type, status=status
        ).inc()

        if size is not None:
            storage_size.labels(storage_type=storage_type).set(size)

    # Error metrics methods

    def record_error(self, error_type: str, component: str, error: Exception | None = None) -> None:
        """Record error occurrence.

        Args:
            error_type: Type of error
            component: Component where error occurred
            error: Optional exception object
        """
        error_counter.labels(error_type=error_type, component=component).inc()

        if error:
            logger.error(
                "error_recorded",
                error_type=error_type,
                component=component,
                error=str(error),
                traceback=traceback.format_exc(),
            )

    # Performance metrics methods

    def record_latency(self, operation: str, duration: float) -> None:
        """Record operation latency.

        Args:
            operation: Operation name
            duration: Duration in seconds
        """
        operation_latency.labels(operation=operation).observe(duration)

    # Custom metrics

    def register_custom_metric(
        self,
        name: str,
        metric_type: str,
        description: str,
        labels: list[str] | None = None,
    ) -> Any:
        """Register custom metric.

        Args:
            name: Metric name
            metric_type: Type (counter, gauge, histogram, summary)
            description: Metric description
            labels: Optional label names

        Returns:
            Prometheus metric object
        """
        if name in self._custom_metrics:
            return self._custom_metrics[name]

        labels = labels or []

        metric: Counter | Gauge | Histogram | Summary
        if metric_type == "counter":
            metric = Counter(name, description, labels)
        elif metric_type == "gauge":
            metric = Gauge(name, description, labels)
        elif metric_type == "histogram":
            metric = Histogram(name, description, labels)
        elif metric_type == "summary":
            metric = Summary(name, description, labels)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

        self._custom_metrics[name] = metric
        return metric

    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in text format.

        Returns:
            Metrics in Prometheus text format
        """
        return cast(bytes, generate_latest(REGISTRY))

    def get_uptime(self) -> float:
        """Get uptime in seconds.

        Returns:
            Uptime in seconds
        """
        return time.time() - self.start_time


class HealthCheck:
    """Simple health check system for monitoring.

    Features:
        - Component health checks
        - Dependency checks
        - Aggregated health status
    """

    def __init__(self) -> None:
        """Initialize health check."""
        self.checks: dict[str, HealthCheckInfo] = {}
        self.last_results: dict[str, dict[str, Any]] = {}

    def register_check(
        self, name: str, check_func: Callable[[], bool], critical: bool = False
    ) -> None:
        """Register a health check.

        Args:
            name: Check name
            check_func: Function that returns True if healthy
            critical: Whether this is a critical check
        """
        self.checks[name] = {"func": check_func, "critical": critical}
        logger.debug("health_check_registered", name=name, critical=critical)

    def run_checks(self) -> dict[str, Any]:
        """Run all health checks.

        Returns:
            Health check results
        """
        results: dict[str, Any] = {
            "status": "healthy",
            "timestamp": utc_now().isoformat(),
            "checks": {},
            "uptime": metrics_collector.get_uptime(),
        }

        has_critical_failure = False
        has_failure = False

        for name, check_info in self.checks.items():
            try:
                is_healthy = check_info["func"]()
                results["checks"][name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "critical": check_info["critical"],
                    "timestamp": time.time(),
                }

                if not is_healthy:
                    has_failure = True
                    if check_info["critical"]:
                        has_critical_failure = True

            except Exception as e:
                results["checks"][name] = {
                    "status": "error",
                    "critical": check_info["critical"],
                    "error": str(e),
                    "timestamp": time.time(),
                }
                has_failure = True
                if check_info["critical"]:
                    has_critical_failure = True

                logger.error("health_check_failed", name=name, error=str(e))

        # Update overall status
        if has_critical_failure:
            results["status"] = "critical"
        elif has_failure:
            results["status"] = "degraded"

        self.last_results = results
        return results

    def get_status(self) -> str:
        """Get overall health status.

        Returns:
            Status string (healthy, degraded, critical)
        """
        if not self.last_results:
            self.run_checks()
        status = self.last_results.get("status", "unknown")
        return str(status) if status is not None else "unknown"

    def is_healthy(self) -> bool:
        """Check if system is healthy.

        Returns:
            True if healthy
        """
        return self.get_status() == "healthy"


# Create global instances
metrics_collector = MetricsCollector()
health_check = HealthCheck()


# Register default health checks
def check_memory() -> bool:
    """Check if memory usage is acceptable."""
    process = psutil.Process()
    memory_mb = cast(float, process.memory_info().rss / 1024 / 1024)
    return memory_mb < 1000  # Less than 1GB


def check_cpu() -> bool:
    """Check if CPU usage is acceptable."""
    cpu_percent = cast(float, psutil.cpu_percent(interval=0.1))
    return cpu_percent < 90


def check_disk() -> bool:
    """Check if disk space is available."""
    settings = get_settings()
    usage = psutil.disk_usage(str(settings.dataset.path))
    return cast(bool, usage.percent < 90)


# Register default checks
health_check.register_check("memory", check_memory)
health_check.register_check("cpu", check_cpu)
health_check.register_check("disk", check_disk, critical=True)
