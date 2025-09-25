"""Monitor management package - ported from Qontinui framework.

This package handles multi-monitor support for the automation framework.
"""

from .monitor_manager import MonitorInfo, MonitorManager

__all__ = [
    "MonitorManager",
    "MonitorInfo",
]
