"""Aspect annotations - ported from Qontinui framework.

Provides decorators for cross-cutting concerns like
monitoring, recovery, and data collection.
"""

from .monitored import (
    monitored,
    is_monitored,
    get_monitored_config,
    MonitoredConfig
)

from .recoverable import (
    recoverable,
    is_recoverable,
    get_recoverable_config
)

from .collect_data import (
    collect_data,
    is_collecting_data,
    get_collect_data_config,
    get_collected_data,
    clear_collected_data,
    CollectedData
)

__all__ = [
    # Monitoring
    'monitored',
    'is_monitored',
    'get_monitored_config',
    'MonitoredConfig',
    
    # Recovery
    'recoverable',
    'is_recoverable',
    'get_recoverable_config',
    
    # Data collection
    'collect_data',
    'is_collecting_data',
    'get_collect_data_config',
    'get_collected_data',
    'clear_collected_data',
    'CollectedData',
]