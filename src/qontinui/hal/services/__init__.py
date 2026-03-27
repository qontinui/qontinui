"""HAL services — higher-level capture orchestration built on HAL interfaces."""

from qontinui.hal.services.event_monitor import EventMonitor, WindowInfo
from qontinui.hal.services.paired_capture import PairedCaptureResult, PairedCaptureService

__all__ = [
    "EventMonitor",
    "PairedCaptureResult",
    "PairedCaptureService",
    "WindowInfo",
]
