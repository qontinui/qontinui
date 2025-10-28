"""Factory methods for Region creation.

Provides various ways to construct Region instances.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .location import Location
    from .region import Region


class RegionFactory:
    """Factory for creating Region instances following Single Responsibility Principle."""

    @staticmethod
    def from_xywh(x: int, y: int, w: int, h: int) -> Region:
        """Create Region from x, y, width, height.

        Args:
            x: X coordinate
            y: Y coordinate
            w: Width
            h: Height

        Returns:
            New Region instance
        """
        from .region import Region

        return Region(x=x, y=y, width=w, height=h)

    @staticmethod
    def from_bounds(x1: int, y1: int, x2: int, y2: int) -> Region:
        """Create Region from bounding coordinates.

        Args:
            x1: Left x coordinate
            y1: Top y coordinate
            x2: Right x coordinate
            y2: Bottom y coordinate

        Returns:
            New Region instance
        """
        from .region import Region

        return Region(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    @staticmethod
    def from_locations(loc1: Location, loc2: Location) -> Region:
        """Create Region as bounding box of two locations.

        Args:
            loc1: First location
            loc2: Second location

        Returns:
            New Region containing both locations
        """
        x1 = min(loc1.x, loc2.x)
        y1 = min(loc1.y, loc2.y)
        x2 = max(loc1.x, loc2.x)
        y2 = max(loc1.y, loc2.y)
        return RegionFactory.from_bounds(x1, y1, x2, y2)

    @staticmethod
    def get_screen_dimensions() -> tuple[int, int]:
        """Get screen dimensions.

        Returns:
            (width, height) tuple
        """
        # Try environment variables first
        width = int(os.environ.get("SCREEN_WIDTH", 0))
        height = int(os.environ.get("SCREEN_HEIGHT", 0))

        if width > 0 and height > 0:
            return width, height

        # Try tkinter
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            if width > 0 and height > 0:
                return width, height
        except (OSError, RuntimeError, ValueError, ImportError, Exception):
            # OK to fallback to default resolution (includes TclError and other display errors)
            pass

        # Default fallback
        return 1920, 1080
