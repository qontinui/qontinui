"""Coordinate service for multi-monitor coordinate translations.

This module provides the CoordinateService singleton, which is the single source
of truth for all coordinate translations in Qontinui. It handles conversions
between different coordinate systems:

- ScreenPoint: Absolute screen coordinates (pyautogui uses these)
- VirtualPoint: Relative to virtual desktop origin (FIND results use these)
- MonitorPoint: Relative to a specific monitor's origin

Usage:
    >>> from qontinui.coordinates import CoordinateService
    >>> service = CoordinateService.get_instance()
    >>>
    >>> # Convert FIND result to screen coordinates for clicking
    >>> screen_point = service.match_to_screen(match_x=65, match_y=1372)
    >>> print(f"Click at ({screen_point.x}, {screen_point.y})")
"""

import threading

from .types import MonitorPoint, ScreenPoint, VirtualPoint
from .virtual_desktop import VirtualDesktopInfo


class CoordinateService:
    """Single source of truth for all coordinate translations.

    This is a thread-safe singleton that provides coordinate conversion
    utilities for multi-monitor automation. It automatically detects the
    current monitor configuration and provides methods to convert between
    different coordinate systems.

    The service caches virtual desktop information and can be refreshed
    when monitor configuration changes (e.g., displays are added/removed).

    Example:
        >>> service = CoordinateService.get_instance()
        >>>
        >>> # Get virtual desktop info
        >>> vd = service.get_virtual_desktop()
        >>> print(f"Virtual desktop: {vd.width}x{vd.height}")
        >>> print(f"Origin: ({vd.origin_x}, {vd.origin_y})")
        >>>
        >>> # Convert FIND match to screen coordinates
        >>> screen_point = service.match_to_screen(100, 200)
        >>>
        >>> # Find which monitor contains a point
        >>> monitor_index = service.get_monitor_at_point(screen_point.x, screen_point.y)
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize CoordinateService.

        Private constructor - use get_instance() instead.
        """
        self._virtual_desktop: VirtualDesktopInfo | None = None
        self._refresh()

    @classmethod
    def get_instance(cls) -> "CoordinateService":
        """Get or create the singleton CoordinateService instance.

        Thread-safe singleton access using double-checked locking.

        Returns:
            The singleton CoordinateService instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_virtual_desktop(self) -> VirtualDesktopInfo:
        """Get current virtual desktop information.

        Returns:
            VirtualDesktopInfo with current monitor configuration

        Raises:
            RuntimeError: If virtual desktop info is not available
        """
        if self._virtual_desktop is None:
            raise RuntimeError("Virtual desktop info not initialized")
        return self._virtual_desktop

    def refresh(self) -> None:
        """Refresh monitor information.

        Call this method when monitor configuration changes (e.g., displays
        are added, removed, or rearranged). This will re-detect all monitors
        and update the virtual desktop bounds.

        Example:
            >>> service = CoordinateService.get_instance()
            >>> # User plugs in a new monitor
            >>> service.refresh()  # Re-detect monitors
        """
        self._refresh()

    def _refresh(self) -> None:
        """Internal refresh implementation."""
        # Use MSSScreenCapture to get current monitor configuration
        # We need to create a temporary instance to access monitors
        import mss

        with mss.mss() as sct:
            self._virtual_desktop = VirtualDesktopInfo.from_mss_monitors(sct.monitors)

    def match_to_screen(self, match_x: int, match_y: int) -> ScreenPoint:
        """Convert FIND result coordinates to absolute screen coordinates.

        FIND captures the entire virtual desktop (MSS monitors[0]). Match
        coordinates are relative to the virtual desktop origin (min_x, min_y).

        This method adds the virtual desktop origin offset to convert from
        virtual-relative coordinates to absolute screen coordinates that
        pyautogui can use.

        Args:
            match_x: X coordinate from FIND result (virtual-relative)
            match_y: Y coordinate from FIND result (virtual-relative)

        Returns:
            ScreenPoint with absolute screen coordinates

        Example:
            >>> # Virtual desktop origin is (-1920, 0)
            >>> # FIND match at (65, 1372) in the screenshot
            >>> screen_point = service.match_to_screen(65, 1372)
            >>> print(screen_point)  # ScreenPoint(x=-1855, y=1372)
            >>> # Now you can click at (screen_point.x, screen_point.y)
        """
        vd = self.get_virtual_desktop()
        return ScreenPoint(
            x=match_x + vd.origin_x,
            y=match_y + vd.origin_y,
        )

    def screen_to_match(self, screen_x: int, screen_y: int) -> VirtualPoint:
        """Convert absolute screen coordinates to virtual-relative coordinates.

        This is the inverse of match_to_screen(). Use this when you have
        absolute screen coordinates and need to find where they would appear
        in a virtual desktop screenshot.

        Args:
            screen_x: X coordinate in absolute screen space
            screen_y: Y coordinate in absolute screen space

        Returns:
            VirtualPoint relative to virtual desktop origin

        Example:
            >>> # Absolute screen point at (-1855, 1372)
            >>> virtual_point = service.screen_to_match(-1855, 1372)
            >>> print(virtual_point)  # VirtualPoint(x=65, y=1372)
            >>> # This would be at pixel (65, 1372) in a FIND screenshot
        """
        vd = self.get_virtual_desktop()
        return VirtualPoint(
            x=screen_x - vd.origin_x,
            y=screen_y - vd.origin_y,
        )

    def monitor_to_screen(self, x: int, y: int, monitor_index: int) -> ScreenPoint:
        """Convert monitor-relative coordinates to absolute screen coordinates.

        Takes a point that's relative to a specific monitor's top-left corner
        and converts it to absolute screen coordinates.

        Args:
            x: X coordinate relative to monitor origin
            y: Y coordinate relative to monitor origin
            monitor_index: 0-based monitor index

        Returns:
            ScreenPoint with absolute screen coordinates

        Raises:
            ValueError: If monitor_index is invalid

        Example:
            >>> # Point at (100, 100) relative to monitor 1
            >>> screen_point = service.monitor_to_screen(100, 100, monitor_index=1)
        """
        vd = self.get_virtual_desktop()
        monitor = vd.get_monitor(monitor_index)

        if monitor is None:
            raise ValueError(
                f"Invalid monitor index: {monitor_index}. "
                f"Available monitors: 0-{len(vd.monitors) - 1}"
            )

        return ScreenPoint(
            x=x + monitor.x,
            y=y + monitor.y,
        )

    def screen_to_monitor(self, screen_x: int, screen_y: int, monitor_index: int) -> MonitorPoint:
        """Convert absolute screen coordinates to monitor-relative coordinates.

        Takes an absolute screen point and converts it to coordinates relative
        to a specific monitor's top-left corner.

        Args:
            screen_x: X coordinate in absolute screen space
            screen_y: Y coordinate in absolute screen space
            monitor_index: 0-based monitor index

        Returns:
            MonitorPoint with monitor-relative coordinates

        Raises:
            ValueError: If monitor_index is invalid

        Example:
            >>> # Absolute screen point
            >>> monitor_point = service.screen_to_monitor(-1855, 1372, monitor_index=0)
            >>> print(monitor_point)  # MonitorPoint(x=65, y=670, monitor=0)
        """
        vd = self.get_virtual_desktop()
        monitor = vd.get_monitor(monitor_index)

        if monitor is None:
            raise ValueError(
                f"Invalid monitor index: {monitor_index}. "
                f"Available monitors: 0-{len(vd.monitors) - 1}"
            )

        return MonitorPoint(
            x=screen_x - monitor.x,
            y=screen_y - monitor.y,
            monitor_index=monitor_index,
        )

    def get_monitor_at_point(self, screen_x: int, screen_y: int) -> int | None:
        """Find which monitor contains the given absolute screen point.

        Args:
            screen_x: X coordinate in absolute screen space
            screen_y: Y coordinate in absolute screen space

        Returns:
            Monitor index (0-based) or None if point is outside all monitors

        Example:
            >>> monitor_idx = service.get_monitor_at_point(-1855, 1372)
            >>> if monitor_idx is not None:
            ...     print(f"Point is on monitor {monitor_idx}")
        """
        vd = self.get_virtual_desktop()

        for monitor in vd.monitors:
            if monitor.contains_point(screen_x, screen_y):
                return monitor.index

        return None

    def get_monitor_count(self) -> int:
        """Get the total number of monitors.

        Returns:
            Number of physical monitors
        """
        vd = self.get_virtual_desktop()
        return len(vd.monitors)

    def to_screen(self, x: int, y: int, monitor_index: int | None = None) -> ScreenPoint:
        """Convert coordinates to absolute screen coordinates.

        This is the primary method for coordinate translation. It handles both
        virtual-desktop-relative coordinates (when monitor_index is None) and
        monitor-relative coordinates (when monitor_index is specified).

        Use this method when translating FIND results to click positions:
        - If FIND captured the entire virtual desktop (monitor_index=None),
          coordinates are relative to the virtual desktop origin.
        - If FIND captured a specific monitor (monitor_index=N),
          coordinates are relative to that monitor's origin.

        Args:
            x: X coordinate to translate
            y: Y coordinate to translate
            monitor_index: If None, treats (x, y) as virtual-desktop-relative.
                          If specified, treats (x, y) as monitor-relative.

        Returns:
            ScreenPoint with absolute screen coordinates for pyautogui/input

        Example:
            >>> service = CoordinateService.get_instance()
            >>>
            >>> # FIND captured virtual desktop - match at (65, 1372)
            >>> screen = service.to_screen(65, 1372)  # Uses virtual desktop origin
            >>>
            >>> # FIND captured monitor 1 - match at (200, 300)
            >>> screen = service.to_screen(200, 300, monitor_index=1)  # Uses monitor offset
        """
        if monitor_index is not None:
            return self.monitor_to_screen(x, y, monitor_index)
        else:
            return self.match_to_screen(x, y)

    def from_screen(
        self, screen_x: int, screen_y: int, monitor_index: int | None = None
    ) -> VirtualPoint | MonitorPoint:
        """Convert absolute screen coordinates to relative coordinates.

        This is the inverse of to_screen(). It converts absolute screen
        coordinates to either virtual-desktop-relative or monitor-relative
        coordinates based on the monitor_index parameter.

        Args:
            screen_x: Absolute screen X coordinate
            screen_y: Absolute screen Y coordinate
            monitor_index: If None, returns virtual-desktop-relative coordinates.
                          If specified, returns monitor-relative coordinates.

        Returns:
            VirtualPoint if monitor_index is None, MonitorPoint otherwise

        Example:
            >>> service = CoordinateService.get_instance()
            >>>
            >>> # Convert to virtual desktop coordinates
            >>> virtual = service.from_screen(-1855, 1372)
            >>> print(f"Virtual: ({virtual.x}, {virtual.y})")
            >>>
            >>> # Convert to monitor 0 coordinates
            >>> monitor = service.from_screen(-1855, 1372, monitor_index=0)
            >>> print(f"Monitor 0: ({monitor.x}, {monitor.y})")
        """
        if monitor_index is not None:
            return self.screen_to_monitor(screen_x, screen_y, monitor_index)
        else:
            return self.screen_to_match(screen_x, screen_y)

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        if self._virtual_desktop is None:
            return "CoordinateService(not initialized)"

        vd = self._virtual_desktop
        return (
            f"CoordinateService("
            f"monitors={len(vd.monitors)}, "
            f"virtual_desktop={vd.width}x{vd.height} "
            f"at ({vd.origin_x}, {vd.origin_y}))"
        )
