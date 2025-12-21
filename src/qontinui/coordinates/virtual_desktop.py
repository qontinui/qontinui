"""Virtual desktop information and calculations.

The virtual desktop is the combined coordinate space spanning all monitors.
This module provides utilities for working with the virtual desktop coordinate
system, which is critical for multi-monitor automation.
"""

from dataclasses import dataclass

from .types import MonitorInfo


@dataclass(frozen=True)
class VirtualDesktopInfo:
    """Immutable representation of the virtual desktop coordinate space.

    The virtual desktop is the bounding box containing all physical monitors.
    Its origin point (origin_x, origin_y) is at (min_x, min_y) across all
    monitors - NOT necessarily at (0, 0).

    This is critical for understanding FIND results: when FIND captures the
    entire virtual desktop (MSS monitors[0]), the resulting screenshot has
    coordinates relative to (origin_x, origin_y), not (0, 0).

    Example:
        Monitor layout:
            Left: x=-1920, y=702, 1920x1080
            Primary: x=0, y=0, 3840x2160
            Right: x=3840, y=702, 1920x1080

        Virtual desktop:
            origin_x = -1920  # min X across all monitors
            origin_y = 0      # min Y across all monitors
            width = 7680      # -1920 to 5760
            height = 2160     # 0 to 2160

        Note: The virtual desktop origin is NOT the left monitor's position!
        It's calculated as (min_x, min_y) across ALL monitors.

    Attributes:
        origin_x: Virtual desktop origin X (min X across all monitors)
        origin_y: Virtual desktop origin Y (min Y across all monitors)
        width: Total virtual desktop width in pixels
        height: Total virtual desktop height in pixels
        monitors: Tuple of all physical monitors (immutable)
    """

    origin_x: int
    origin_y: int
    width: int
    height: int
    monitors: tuple[MonitorInfo, ...]

    @classmethod
    def from_mss_monitors(cls, mss_monitors: list[dict[str, int]]) -> "VirtualDesktopInfo":
        """Create VirtualDesktopInfo from MSS monitor list.

        MSS provides a special monitor list where:
            - monitors[0] = Virtual desktop (all monitors combined)
            - monitors[1..n] = Physical monitors

        IMPORTANT: The virtual desktop origin is calculated from ALL physical
        monitors, NOT just taken from monitors[0]. This is because MSS might
        not always calculate it correctly for all multi-monitor configurations.

        Args:
            mss_monitors: List of monitors from mss.mss().monitors

        Returns:
            VirtualDesktopInfo with correct origin and bounds

        Example:
            >>> import mss
            >>> with mss.mss() as sct:
            ...     vd_info = VirtualDesktopInfo.from_mss_monitors(sct.monitors)
            >>> print(f"Virtual desktop origin: ({vd_info.origin_x}, {vd_info.origin_y})")
        """
        # Physical monitors start at index 1 (skip virtual desktop at index 0)
        physical_monitors = mss_monitors[1:]

        if not physical_monitors:
            # No monitors found - create default
            return cls(
                origin_x=0,
                origin_y=0,
                width=1920,
                height=1080,
                monitors=(),
            )

        # Calculate virtual desktop origin as (min_x, min_y) across ALL monitors
        min_x = min(mon["left"] for mon in physical_monitors)
        min_y = min(mon["top"] for mon in physical_monitors)
        max_x = max(mon["left"] + mon["width"] for mon in physical_monitors)
        max_y = max(mon["top"] + mon["height"] for mon in physical_monitors)

        # Build MonitorInfo objects (0-based indexing)
        monitor_infos = []
        for i, mon in enumerate(physical_monitors):
            info = MonitorInfo(
                index=i,  # 0-based index
                x=mon["left"],
                y=mon["top"],
                width=mon["width"],
                height=mon["height"],
                is_primary=(i == 0),  # First monitor is typically primary
            )
            monitor_infos.append(info)

        return cls(
            origin_x=min_x,
            origin_y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
            monitors=tuple(monitor_infos),
        )

    def get_monitor(self, index: int) -> MonitorInfo | None:
        """Get monitor by index.

        Args:
            index: 0-based monitor index

        Returns:
            MonitorInfo or None if index is invalid
        """
        if 0 <= index < len(self.monitors):
            return self.monitors[index]
        return None

    def get_primary_monitor(self) -> MonitorInfo | None:
        """Get the primary monitor.

        Returns:
            Primary MonitorInfo or first monitor if no primary is marked
        """
        for monitor in self.monitors:
            if monitor.is_primary:
                return monitor

        # Fallback to first monitor
        return self.monitors[0] if self.monitors else None

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"VirtualDesktopInfo("
            f"origin=({self.origin_x}, {self.origin_y}), "
            f"size=({self.width}x{self.height}), "
            f"monitors={len(self.monitors)})"
        )
