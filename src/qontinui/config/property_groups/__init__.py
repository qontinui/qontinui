"""Property groups for Qontinui configuration.

This package organizes configuration properties into themed groups:
- core_properties: Core application settings
- input_properties: Mouse/keyboard settings
- vision_properties: Vision/image finding settings
- timing_properties: Timing/wait settings
- output_properties: Screenshot/recording/dataset settings
- logging_properties: Logging settings
- debug_properties: Debug/testing settings
- display_properties: Visual/monitor/capture settings
"""

from .core_properties import CoreProperties
from .debug_properties import DebugProperties
from .display_properties import DisplayProperties
from .input_properties import InputProperties
from .logging_properties import LoggingProperties
from .output_properties import OutputProperties
from .timing_properties import TimingProperties
from .vision_properties import VisionProperties

__all__ = [
    "CoreProperties",
    "DebugProperties",
    "DisplayProperties",
    "InputProperties",
    "LoggingProperties",
    "OutputProperties",
    "TimingProperties",
    "VisionProperties",
]
