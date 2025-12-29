"""Coordinate type definitions for multi-monitor support.

This module defines immutable coordinate types for different coordinate systems
used in Qontinui's multi-monitor automation:

1. ScreenPoint - Absolute screen coordinates (pyautogui uses these)
2. VirtualPoint - Relative to virtual desktop origin (FIND results use these)
3. MonitorPoint - Relative to a specific monitor's origin

These types provide type safety and clarity when working with coordinates
across different systems.

Note: Schema types (CoordinateSystem, Coordinates, Region) are re-exported from
qontinui-schemas in the parent package for configuration purposes. The point types
here are used internally for coordinate translation operations.
"""

from dataclasses import dataclass

# Re-export schema types for coordinate configuration
# These are Pydantic models used in configuration files
from qontinui_schemas.config.models.geometry import (
    Coordinates,
    CoordinateSystem,
    Region,
)

__all__ = [
    # Local point types for coordinate translation
    "ScreenPoint",
    "VirtualPoint",
    "MonitorPoint",
    "MonitorInfo",
    # Schema types for configuration
    "CoordinateSystem",
    "Coordinates",
    "Region",
]


@dataclass(frozen=True)
class ScreenPoint:
    """A point in absolute screen coordinates.

    Screen coordinates are absolute coordinates in the virtual desktop space.
    This is what pyautogui and other input automation libraries use.

    On multi-monitor systems, these coordinates can be negative for monitors
    positioned left of or above the primary monitor.

    Example:
        Monitor layout:
            Left: x=-1920, y=0, 1920x1080
            Primary: x=0, y=0, 1920x1080

        A point on the left monitor at its center would be:
            ScreenPoint(x=-960, y=540)

    Attributes:
        x: X coordinate in absolute screen space
        y: Y coordinate in absolute screen space
    """

    x: int
    y: int

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"ScreenPoint(x={self.x}, y={self.y})"


@dataclass(frozen=True)
class VirtualPoint:
    """A point relative to virtual desktop origin.

    Virtual coordinates are relative to the virtual desktop's origin point,
    which is at (min_x, min_y) across all monitors. FIND captures the entire
    virtual desktop, so match coordinates are VirtualPoints.

    The virtual desktop origin is NOT necessarily (0, 0) - it's the minimum
    X and minimum Y across all physical monitors.

    Example:
        Monitor layout:
            Left: x=-1920, y=702, 1920x1080
            Primary: x=0, y=0, 3840x2160
            Right: x=3840, y=702, 1920x1080

        Virtual desktop origin: (-1920, 0)  # min_x=-1920, min_y=0

        A FIND match at pixel (65, 1372) in the screenshot is:
            VirtualPoint(x=65, y=1372)

        To convert to screen coordinates for clicking:
            ScreenPoint(x=65 + (-1920), y=1372 + 0) = ScreenPoint(-1855, 1372)

    Attributes:
        x: X coordinate relative to virtual desktop origin
        y: Y coordinate relative to virtual desktop origin
    """

    x: int
    y: int

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"VirtualPoint(x={self.x}, y={self.y})"


@dataclass(frozen=True)
class MonitorPoint:
    """A point relative to a specific monitor's origin.

    Monitor coordinates are relative to a specific monitor's top-left corner.
    This is useful for operations that are relative to a particular display.

    Attributes:
        x: X coordinate relative to monitor origin (top-left)
        y: Y coordinate relative to monitor origin (top-left)
        monitor_index: 0-based index of the monitor (0 = first physical monitor)
    """

    x: int
    y: int
    monitor_index: int

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"MonitorPoint(x={self.x}, y={self.y}, monitor={self.monitor_index})"


@dataclass(frozen=True)
class MonitorInfo:
    """Information about a physical monitor.

    Represents a single physical monitor in the system. Used internally
    by CoordinateService to perform coordinate translations.

    Attributes:
        index: 0-based monitor index (0 = first physical monitor)
        x: Monitor X position in absolute screen coordinates
        y: Monitor Y position in absolute screen coordinates
        width: Monitor width in pixels
        height: Monitor height in pixels
        is_primary: True if this is the primary monitor
    """

    index: int
    x: int
    y: int
    width: int
    height: int
    is_primary: bool

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Get monitor bounds as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    def contains_point(self, x: int, y: int) -> bool:
        """Check if absolute screen point is within monitor bounds.

        Args:
            x: X coordinate in absolute screen space
            y: Y coordinate in absolute screen space

        Returns:
            True if point is within this monitor
        """
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        primary_str = " (primary)" if self.is_primary else ""
        return (
            f"MonitorInfo(index={self.index}, "
            f"x={self.x}, y={self.y}, "
            f"width={self.width}, height={self.height}"
            f"{primary_str})"
        )
