"""Coordinate system abstractions for multi-monitor automation.

This module provides clean, type-safe abstractions for working with coordinates
across different coordinate systems in multi-monitor setups.

## Coordinate Systems

Qontinui uses three coordinate systems:

1. **ScreenPoint** - Absolute screen coordinates
   - What pyautogui and input automation libraries use
   - Can be negative for monitors left of/above primary monitor
   - Example: `ScreenPoint(x=-1855, y=1372)`

2. **VirtualPoint** - Relative to virtual desktop origin
   - What FIND results use (screenshot pixel coordinates)
   - Relative to (min_x, min_y) across all monitors
   - Example: `VirtualPoint(x=65, y=1372)`

3. **MonitorPoint** - Relative to a specific monitor's origin
   - Useful for monitor-specific operations
   - Always has a monitor_index
   - Example: `MonitorPoint(x=100, y=200, monitor=1)`

## Schema Types

The package also re-exports schema types from qontinui-schemas for configuration:

- `CoordinateSystem` - Enum for coordinate system identification
- `Coordinates` - Pydantic model for x,y coordinates with optional system
- `Region` - Pydantic model for rectangular regions
- `Monitor` (as SchemaMonitor) - Pydantic model for monitor configuration
- `VirtualDesktop` (as SchemaVirtualDesktop) - Pydantic model for virtual desktop

## Virtual Desktop

The virtual desktop is the bounding box containing all monitors. Its origin
is at (min_x, min_y) across all physical monitors - NOT necessarily (0, 0).

When FIND captures the entire virtual desktop, match coordinates are relative
to this origin point.

## Usage

### Basic Coordinate Conversion

```python
from qontinui.coordinates import CoordinateService

# Get the singleton service
service = CoordinateService.get_instance()

# Convert FIND match to screen coordinates for clicking
screen_point = service.match_to_screen(match_x=65, match_y=1372)
print(f"Click at ({screen_point.x}, {screen_point.y})")

# Find which monitor contains a point
monitor_idx = service.get_monitor_at_point(screen_point.x, screen_point.y)
print(f"Point is on monitor {monitor_idx}")
```

### Virtual Desktop Information

```python
# Get virtual desktop info
vd = service.get_virtual_desktop()
print(f"Virtual desktop: {vd.width}x{vd.height}")
print(f"Origin: ({vd.origin_x}, {vd.origin_y})")
print(f"Monitors: {len(vd.monitors)}")

# Access monitor information
for monitor in vd.monitors:
    print(f"Monitor {monitor.index}: {monitor.width}x{monitor.height} at ({monitor.x}, {monitor.y})")
```

### Monitor Configuration Changes

```python
# Call this when displays are added/removed/rearranged
service.refresh()
```

## Example: Multi-Monitor Setup

```
Monitor layout:
    Left:    x=-1920, y=702,  1920x1080
    Primary: x=0,     y=0,    3840x2160
    Right:   x=3840,  y=702,  1920x1080

Virtual desktop:
    origin: (-1920, 0)     # min_x, min_y across all monitors
    size:   (7680, 2160)   # from -1920 to 5760, from 0 to 2160

FIND match at (65, 1372) in screenshot:
    VirtualPoint(x=65, y=1372)

Convert to screen coordinates:
    ScreenPoint(x=65 + (-1920), y=1372 + 0) = ScreenPoint(-1855, 1372)

Now pyautogui can click at (-1855, 1372) to click the match!
```

## Exports

This package exports:
- `ScreenPoint` - Absolute screen coordinates type
- `VirtualPoint` - Virtual desktop relative coordinates type
- `MonitorPoint` - Monitor-relative coordinates type
- `MonitorInfo` - Monitor information type (local dataclass)
- `VirtualDesktopInfo` - Virtual desktop information (local dataclass)
- `CoordinateService` - Singleton service for coordinate translations
- `CoordinateSystem` - Schema enum for coordinate system types
- `Coordinates` - Schema model for x,y coordinates
- `Region` - Schema model for rectangular regions
- `SchemaMonitor` - Schema model for monitor configuration (alias for Monitor)
- `SchemaVirtualDesktop` - Schema model for virtual desktop (alias for VirtualDesktop)
"""

from .service import CoordinateService
from .types import (
    Coordinates,
    CoordinateSystem,
    MonitorInfo,
    MonitorPoint,
    Region,
    ScreenPoint,
    VirtualPoint,
)
from .virtual_desktop import SchemaMonitor, SchemaVirtualDesktop, VirtualDesktopInfo

__all__ = [
    # Local point types for coordinate translation
    "ScreenPoint",
    "VirtualPoint",
    "MonitorPoint",
    "MonitorInfo",
    # Virtual Desktop
    "VirtualDesktopInfo",
    # Service
    "CoordinateService",
    # Schema types (from qontinui-schemas)
    "CoordinateSystem",
    "Coordinates",
    "Region",
    "SchemaMonitor",
    "SchemaVirtualDesktop",
]
