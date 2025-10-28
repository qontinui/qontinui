"""
Base types and enums used across action configurations.

This module provides fundamental enumerations and primitive types that are
referenced throughout the action schema system.
"""

from enum import Enum


class MouseButton(str, Enum):
    """Mouse button types."""

    LEFT = "LEFT"
    RIGHT = "RIGHT"
    MIDDLE = "MIDDLE"


class SearchStrategy(str, Enum):
    """Search strategy for finding targets."""

    FIRST = "FIRST"
    ALL = "ALL"
    BEST = "BEST"
    EACH = "EACH"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class VerificationMode(str, Enum):
    """Verification modes for action results."""

    IMAGE_APPEARS = "IMAGE_APPEARS"
    IMAGE_DISAPPEARS = "IMAGE_DISAPPEARS"
    TEXT_APPEARS = "TEXT_APPEARS"
    TEXT_DISAPPEARS = "TEXT_DISAPPEARS"
    STATE_CHANGE = "STATE_CHANGE"
    NONE = "NONE"


class WorkflowVisibility(str, Enum):
    """Workflow visibility levels for UI filtering.

    - PUBLIC: Normal workflows visible in UI dropdowns and lists
    - INTERNAL: Inline/helper workflows hidden from UI (but executable)
    - SYSTEM: Framework-generated workflows for internal use
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    SYSTEM = "system"
