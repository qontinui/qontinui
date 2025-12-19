# Coordinates Module

Clean abstractions for coordinate handling in multi-monitor automation.

## Overview

This module provides type-safe coordinate abstractions for working with different coordinate systems in multi-monitor setups. It eliminates confusion between absolute screen coordinates, virtual desktop coordinates, and monitor-relative coordinates.

## Coordinate Systems

### 1. ScreenPoint - Absolute Screen Coordinates

What pyautogui and input automation libraries use. Can be negative for monitors positioned left of or above the primary monitor.

```python
from qontinui.coordinates import ScreenPoint

point = ScreenPoint(x=-1855, y=1372)  # Point on left monitor
```

### 2. VirtualPoint - Virtual Desktop Relative

What FIND results use. Relative to the virtual desktop origin (min_x, min_y across all monitors).

```python
from qontinui.coordinates import VirtualPoint

point = VirtualPoint(x=65, y=1372)  # Pixel (65, 1372) in FIND screenshot
```

### 3. MonitorPoint - Monitor Relative

Relative to a specific monitor's top-left corner. Useful for monitor-specific operations.

```python
from qontinui.coordinates import MonitorPoint

point = MonitorPoint(x=100, y=200, monitor_index=1)  # On monitor 1
```

## Virtual Desktop

The virtual desktop is the bounding box containing all monitors. Its origin is at (min_x, min_y) across all physical monitors - **NOT necessarily (0, 0)**.

Example monitor layout:
```
Left Monitor:    x=-1920, y=702,  1920x1080
Primary Monitor: x=0,     y=0,    3840x2160
Right Monitor:   x=3840,  y=702,  1920x1080
```

Virtual desktop:
- Origin: `(-1920, 0)` - minimum X and Y across all monitors
- Size: `7680x2160` - spans from -1920 to 5760 horizontally
- Contains all three monitors

## Usage

### Basic Setup

```python
from qontinui.coordinates import CoordinateService

# Get singleton instance (thread-safe)
service = CoordinateService.get_instance()

# Get virtual desktop information
vd = service.get_virtual_desktop()
print(f"Virtual desktop: {vd.width}x{vd.height} at ({vd.origin_x}, {vd.origin_y})")
print(f"Monitors: {len(vd.monitors)}")
```

### Converting FIND Results to Screen Coordinates

The most common use case - convert FIND match coordinates to screen coordinates for clicking:

```python
# FIND returns a match at pixel (65, 1372) in the virtual desktop screenshot
match_x, match_y = 65, 1372

# Convert to absolute screen coordinates
screen_point = service.match_to_screen(match_x, match_y)

# Now you can click using pyautogui
import pyautogui
pyautogui.click(screen_point.x, screen_point.y)
```

### Finding Which Monitor Contains a Point

```python
# Absolute screen coordinates
screen_x, screen_y = -1855, 1372

# Find which monitor contains this point
monitor_idx = service.get_monitor_at_point(screen_x, screen_y)

if monitor_idx is not None:
    monitor = vd.get_monitor(monitor_idx)
    print(f"Point is on monitor {monitor_idx}: {monitor.width}x{monitor.height}")
else:
    print("Point is outside all monitors")
```

### Monitor-Relative Coordinates

```python
# Convert monitor-relative to screen coordinates
screen_point = service.monitor_to_screen(x=100, y=100, monitor_index=1)

# Convert screen coordinates to monitor-relative
monitor_point = service.screen_to_monitor(screen_x=-1820, screen_y=802, monitor_index=1)
print(f"On monitor {monitor_point.monitor_index} at ({monitor_point.x}, {monitor_point.y})")
```

### Handling Monitor Configuration Changes

Call `refresh()` when displays are added, removed, or rearranged:

```python
# User plugs in a new monitor or changes layout
service.refresh()

# Virtual desktop info is now updated
vd = service.get_virtual_desktop()
```

## Integration with Existing Code

### Replacing Manual Offset Calculations

**Before:**
```python
# Manual offset calculation (error-prone)
offset_x = getattr(context, "monitor_offset_x", 0)
offset_y = getattr(context, "monitor_offset_y", 0)
final_x = match.x + offset_x
final_y = match.y + offset_y
```

**After:**
```python
# Clean, typed conversion
from qontinui.coordinates import CoordinateService

service = CoordinateService.get_instance()
screen_point = service.match_to_screen(match.x, match.y)
final_x, final_y = screen_point.x, screen_point.y
```

### In Mouse Action Executor

```python
from qontinui.coordinates import CoordinateService

class MouseActionExecutor:
    def __init__(self):
        self.coord_service = CoordinateService.get_instance()

    def _get_target_location(self, target):
        # ... existing logic ...

        if isinstance(target, LastFindResultTarget):
            if self.context.last_action_result and self.context.last_action_result.matches:
                match = self.context.last_action_result.matches[0]

                # Clean conversion using coordinates service
                screen_point = self.coord_service.match_to_screen(match.x, match.y)
                return (screen_point.x, screen_point.y)
```

## API Reference

### CoordinateService

**Singleton Methods:**
- `get_instance()` - Get or create the singleton instance

**Instance Methods:**
- `get_virtual_desktop()` - Get current virtual desktop information
- `refresh()` - Refresh monitor information (call when displays change)
- `match_to_screen(match_x, match_y)` - Convert FIND result to screen coordinates
- `screen_to_match(screen_x, screen_y)` - Convert screen to virtual-relative coordinates
- `monitor_to_screen(x, y, monitor_index)` - Convert monitor-relative to screen
- `screen_to_monitor(screen_x, screen_y, monitor_index)` - Convert screen to monitor-relative
- `get_monitor_at_point(screen_x, screen_y)` - Find which monitor contains a point
- `get_monitor_count()` - Get total number of monitors

### VirtualDesktopInfo

**Attributes:**
- `origin_x` - Virtual desktop origin X (min X across all monitors)
- `origin_y` - Virtual desktop origin Y (min Y across all monitors)
- `width` - Total virtual desktop width
- `height` - Total virtual desktop height
- `monitors` - Tuple of MonitorInfo objects

**Methods:**
- `get_monitor(index)` - Get monitor by index
- `get_primary_monitor()` - Get the primary monitor

### MonitorInfo

**Attributes:**
- `index` - 0-based monitor index
- `x` - Monitor X position in absolute screen coordinates
- `y` - Monitor Y position in absolute screen coordinates
- `width` - Monitor width in pixels
- `height` - Monitor height in pixels
- `is_primary` - True if this is the primary monitor

**Properties:**
- `bounds` - Tuple of (x, y, width, height)

**Methods:**
- `contains_point(x, y)` - Check if absolute screen point is within monitor

## Implementation Details

### Thread Safety

CoordinateService is a thread-safe singleton using double-checked locking:

```python
@classmethod
def get_instance(cls):
    if cls._instance is None:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
    return cls._instance
```

### Immutability

All coordinate types (ScreenPoint, VirtualPoint, MonitorPoint) are frozen dataclasses, preventing accidental mutation:

```python
@dataclass(frozen=True)
class ScreenPoint:
    x: int
    y: int
```

### Virtual Desktop Origin Calculation

The virtual desktop origin is calculated as (min_x, min_y) across ALL monitors:

```python
min_x = min(mon["left"] for mon in physical_monitors)
min_y = min(mon["top"] for mon in physical_monitors)
```

This is **not** the same as taking the position of a specific monitor!

## Testing

Run tests to verify functionality:

```python
from qontinui.coordinates import CoordinateService

service = CoordinateService.get_instance()

# Test round-trip conversion
match_x, match_y = 100, 200
screen_point = service.match_to_screen(match_x, match_y)
virtual_point = service.screen_to_match(screen_point.x, screen_point.y)

assert virtual_point.x == match_x
assert virtual_point.y == match_y
print("Round-trip conversion test passed!")
```

## Migration Guide

### Step 1: Replace Manual Offset Access

Find all instances of:
```python
offset_x = getattr(self.context, "monitor_offset_x", 0)
offset_y = getattr(self.context, "monitor_offset_y", 0)
```

Replace with:
```python
from qontinui.coordinates import CoordinateService
service = CoordinateService.get_instance()
# Use service.match_to_screen() instead
```

### Step 2: Update Mouse Action Executor

Update `_get_target_location()` to use CoordinateService for all coordinate conversions.

### Step 3: Remove Redundant Offset Propagation

Once all code uses CoordinateService, you can remove the offset propagation from context if desired.

## Future Enhancements

Possible future improvements:
- DPI-aware coordinate conversion
- Rotation-aware coordinates for rotated displays
- Per-monitor DPI scaling support
- Coordinate history/debugging tools
