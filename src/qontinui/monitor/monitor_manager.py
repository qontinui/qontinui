"""Monitor manager - ported from Qontinui framework.

Manages multi-monitor support for the automation framework.
"""

import logging
import tkinter as tk
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MonitorInfo:
    """Information about a monitor.

    Port of MonitorInfo from Qontinui framework inner class.
    """

    index: int
    """Monitor index."""

    x: int
    """Monitor X position."""

    y: int
    """Monitor Y position."""

    width: int
    """Monitor width."""

    height: int
    """Monitor height."""

    device_id: str
    """Device identifier."""

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Get monitor bounds as tuple.

        Returns:
            (x, y, width, height) tuple
        """
        return (self.x, self.y, self.width, self.height)

    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is within monitor bounds.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is within bounds
        """
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height


class MonitorManager:
    """Manages multi-monitor support for automation framework.

    Port of MonitorManager from Qontinui framework class.

    Provides methods to detect, select, and work with multiple monitors.
    """

    def __init__(self, properties: Optional["BrobotProperties"] = None) -> None:
        """Initialize MonitorManager.

        Args:
            properties: Framework properties
        """
        self.properties = properties
        self.monitor_cache: dict[int, MonitorInfo] = {}
        self.operation_monitor_map: dict[str, int] = {}
        self.headless_mode = False
        self.primary_monitor_index = 0

        self._initialize_monitors()

        if properties and properties.monitor.operation_monitor_map:
            self.operation_monitor_map.update(properties.monitor.operation_monitor_map)

    def _detect_primary_monitor(self) -> int:
        """Detect the primary monitor based on position.

        Returns:
            Index of primary monitor
        """
        if not self.monitor_cache:
            return 0

        # Find monitor closest to (0,0) - typically the primary
        closest_index = 0
        closest_distance = float("inf")

        for index, info in self.monitor_cache.items():
            # Calculate distance from (0,0)
            distance = (info.x**2 + info.y**2) ** 0.5

            if distance < closest_distance:
                closest_distance = distance
                closest_index = index

        primary_info = self.monitor_cache.get(closest_index)
        if primary_info:
            logger.info(
                f"Primary monitor detected at position ({primary_info.x},{primary_info.y}): "
                f"Monitor {closest_index}"
            )

        return closest_index

    def _initialize_monitors(self) -> None:
        """Initialize monitor information and cache available monitors.

        Uses MSS for multi-monitor detection on all platforms.
        MSS monitors[0] is the virtual desktop, physical monitors start at index 1.
        """
        # Check if headless mode is forced
        import os

        headless_property = os.environ.get("DISPLAY")
        if headless_property is None and os.name != "nt":
            logger.info("Headless mode detected (no DISPLAY environment variable)")
            self.headless_mode = True

        if self.headless_mode:
            logger.warning("Running in headless mode. Monitor detection disabled.")
            # Create a default monitor for headless mode
            info = MonitorInfo(
                index=0, x=0, y=0, width=1920, height=1080, device_id="headless-default"
            )
            self.monitor_cache[0] = info
            return

        try:
            # Use MSS for multi-monitor detection
            import mss

            with mss.mss() as sct:
                # MSS monitors[0] is virtual desktop, physical monitors start at index 1
                # We use 0-based indexing in our cache
                for i, mon in enumerate(sct.monitors[1:], 0):
                    info = MonitorInfo(
                        index=i,
                        x=mon["left"],
                        y=mon["top"],
                        width=mon["width"],
                        height=mon["height"],
                        device_id=f"monitor-{i}",
                    )
                    self.monitor_cache[i] = info

                    if self.properties and self.properties.monitor.log_monitor_info:
                        logger.info(
                            f"Monitor {i}: {info.device_id} - "
                            f"Bounds: x={info.x}, y={info.y}, "
                            f"width={info.width}, height={info.height}"
                        )

            logger.info(f"Detected {len(self.monitor_cache)} monitor(s)")

            # Determine the primary monitor
            self.primary_monitor_index = self._detect_primary_monitor()

        except ImportError:
            logger.warning("MSS not available, falling back to tkinter")
            self._initialize_monitors_tkinter()
        except Exception as e:
            logger.error(f"Error during monitor initialization: {e}. Creating default monitor.")
            info = MonitorInfo(
                index=0, x=0, y=0, width=1920, height=1080, device_id="error-default"
            )
            self.monitor_cache[0] = info

    def _initialize_monitors_tkinter(self) -> None:
        """Fallback monitor initialization using tkinter (single monitor only)."""
        try:
            root = tk.Tk()
            root.withdraw()

            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()

            info = MonitorInfo(
                index=0,
                x=0,
                y=0,
                width=screen_width,
                height=screen_height,
                device_id="primary",
            )
            self.monitor_cache[0] = info

            root.destroy()

            logger.info("Detected 1 monitor(s) (tkinter fallback)")
            self.primary_monitor_index = 0

        except Exception as e:
            logger.error(f"Tkinter fallback failed: {e}. Creating default monitor.")
            info = MonitorInfo(
                index=0, x=0, y=0, width=1920, height=1080, device_id="error-default"
            )
            self.monitor_cache[0] = info

    def get_screen(
        self, monitor_index: int | None = None, operation_name: str | None = None
    ) -> Optional["Screen"]:
        """Get Screen object for specified monitor or operation.

        Args:
            monitor_index: Optional monitor index (0-based)
            operation_name: Optional operation name for specific monitor assignment

        Returns:
            Screen object for the specified monitor or None in headless mode
        """
        if self.headless_mode:
            logger.debug("Running in headless mode - returning None Screen")
            return None

        # Determine which monitor to use
        if operation_name and operation_name in self.operation_monitor_map:
            monitor_index = self.operation_monitor_map[operation_name]
            if self.properties and self.properties.monitor.log_monitor_info:
                logger.debug(f"Operation '{operation_name}' assigned to monitor {monitor_index}")
        elif monitor_index is None:
            # Use default monitor from configuration or primary
            if self.properties and self.properties.monitor.default_screen_index >= 0:
                monitor_index = self.properties.monitor.default_screen_index
            elif self.properties and self.properties.monitor.default_screen_index == -1:
                monitor_index = self.primary_monitor_index
                logger.debug(f"Using detected primary monitor: Monitor {monitor_index}")
            else:
                monitor_index = 0

        if not self.is_valid_monitor_index(monitor_index):
            logger.warning(f"Invalid monitor index: {monitor_index}. Using primary monitor.")
            monitor_index = self.primary_monitor_index

        if self.properties and self.properties.monitor.log_monitor_info:
            info = self.monitor_cache.get(monitor_index)
            if info:
                logger.debug(f"Using monitor {monitor_index}: {info.device_id} for operation")

        # Return a Screen-like object (would need actual implementation)
        # For now, return None to indicate we're not using SikuliX screens
        return None

    def get_all_screens(self) -> list[Optional["Screen"]]:
        """Get all available screens for multi-monitor search.

        Returns:
            List of all Screen objects
        """
        screens: list[Screen | None] = []
        if self.headless_mode:
            logger.debug("Running in headless mode - returning empty screen list")
            return screens

        for i in range(self.get_monitor_count()):
            screens.append(self.get_screen(monitor_index=i))

        return screens

    def is_valid_monitor_index(self, index: int) -> bool:
        """Check if monitor index is valid.

        Args:
            index: Monitor index to check

        Returns:
            True if index is valid
        """
        return 0 <= index < self.get_monitor_count()

    def get_monitor_count(self) -> int:
        """Get total number of monitors.

        Returns:
            Number of monitors
        """
        return len(self.monitor_cache)

    def get_primary_monitor_index(self) -> int:
        """Get the index of the primary monitor.

        Returns:
            Primary monitor index
        """
        return self.primary_monitor_index

    def get_monitor_info(self, index: int) -> MonitorInfo | None:
        """Get monitor information.

        Args:
            index: Monitor index

        Returns:
            MonitorInfo or None
        """
        return self.monitor_cache.get(index)

    def get_all_monitor_info(self) -> list[MonitorInfo]:
        """Get all monitor information.

        Returns:
            List of all MonitorInfo objects
        """
        return list(self.monitor_cache.values())

    def set_operation_monitor(self, operation_name: str, monitor_index: int) -> None:
        """Set monitor for specific operation.

        Args:
            operation_name: Name of operation
            monitor_index: Monitor index to use
        """
        if self.is_valid_monitor_index(monitor_index):
            self.operation_monitor_map[operation_name] = monitor_index
            logger.info(f"Assigned operation '{operation_name}' to monitor {monitor_index}")
        else:
            logger.error(
                f"Cannot assign operation '{operation_name}' to invalid monitor index: "
                f"{monitor_index}"
            )

    def get_monitor_at_point(self, x: int, y: int) -> int:
        """Get the monitor containing a specific point.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Monitor index containing the point
        """
        for info in self.monitor_cache.values():
            if info.contains_point(x, y):
                return info.index
        return 0  # Default to primary if not found

    def to_monitor_coordinates(
        self, global_x: int, global_y: int, monitor_index: int
    ) -> tuple[int, int]:
        """Convert global coordinates to monitor-relative coordinates.

        Args:
            global_x: Global X coordinate
            global_y: Global Y coordinate
            monitor_index: Target monitor index

        Returns:
            (x, y) tuple in monitor coordinates
        """
        info = self.monitor_cache.get(monitor_index)
        if not info:
            return (global_x, global_y)

        return (global_x - info.x, global_y - info.y)

    def to_global_coordinates(
        self, monitor_x: int, monitor_y: int, monitor_index: int
    ) -> tuple[int, int]:
        """Convert monitor-relative coordinates to global coordinates.

        Args:
            monitor_x: Monitor-relative X coordinate
            monitor_y: Monitor-relative Y coordinate
            monitor_index: Source monitor index

        Returns:
            (x, y) tuple in global coordinates
        """
        info = self.monitor_cache.get(monitor_index)
        if not info:
            return (monitor_x, monitor_y)

        return (monitor_x + info.x, monitor_y + info.y)


class BrobotProperties:
    """Placeholder for BrobotProperties.

    Will be implemented when migrating the config package.
    """

    def __init__(self) -> None:
        """Initialize properties."""
        self.monitor = MonitorProperties()


class MonitorProperties:
    """Placeholder for monitor properties."""

    def __init__(self) -> None:
        """Initialize monitor properties."""
        self.multi_monitor_enabled = False
        self.log_monitor_info = False
        self.default_screen_index = 0
        self.operation_monitor_map: dict[str, int] = {}


class Screen:
    """Placeholder for Screen class.

    Would represent a SikuliX Screen or equivalent.
    """

    pass
