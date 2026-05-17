"""Backwards-compatibility shim for property_groups.

The property_group pydantic models now live in qontinui-schemas so they can
be shared between the runner (qontinui) and the web tier (qontinui-web).
This package re-exports the schemas versions for legacy import paths.
"""

from qontinui_schemas.config.property_groups import (  # noqa: F401
    CoreProperties,
    DebugProperties,
    DisplayProperties,
    InputProperties,
    LoggingProperties,
    OutputProperties,
    TimingProperties,
    VisionProperties,
)

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
