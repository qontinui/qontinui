"""Aspect annotations - ported from Qontinui framework.

Provides decorators for cross-cutting concerns like
monitoring, recovery, and data collection.
"""

from .collect_data import (
    CollectedData,
    clear_collected_data,
    collect_data,
    get_collect_data_config,
    get_collected_data,
    is_collecting_data,
)
from .monitored import MonitoredConfig, get_monitored_config, is_monitored, monitored
from .recoverable import get_recoverable_config, is_recoverable, recoverable

__all__ = [
    # Monitoring
    "monitored",
    "is_monitored",
    "get_monitored_config",
    "MonitoredConfig",
    # Recovery
    "recoverable",
    "is_recoverable",
    "get_recoverable_config",
    # Data collection
    "collect_data",
    "is_collecting_data",
    "get_collect_data_config",
    "get_collected_data",
    "clear_collected_data",
    "CollectedData",
]
