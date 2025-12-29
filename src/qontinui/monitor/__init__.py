"""Monitor management package - ported from Qontinui framework.

This package handles multi-monitor support for the automation framework.

Re-exports schema types from qontinui-schemas for configuration purposes:
- SchemaMonitor: Pydantic model for monitor configuration
- SchemaVirtualDesktop: Pydantic model for virtual desktop configuration
"""

from qontinui_schemas.config.models.monitors import Monitor as SchemaMonitor
from qontinui_schemas.config.models.monitors import VirtualDesktop as SchemaVirtualDesktop

from .monitor_manager import MonitorInfo, MonitorManager

__all__ = [
    # Local types
    "MonitorManager",
    "MonitorInfo",
    # Schema types (from qontinui-schemas)
    "SchemaMonitor",
    "SchemaVirtualDesktop",
]
