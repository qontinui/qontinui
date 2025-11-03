"""Highlight overlay implementation for visual feedback.

This module provides a cross-platform overlay window for highlighting screen regions
using tkinter. The overlay is transparent, non-blocking, and automatically dismisses
after a specified duration.
"""

import logging
import threading
import tkinter as tk
from typing import Literal

logger = logging.getLogger(__name__)


class HighlightOverlay:
    """Cross-platform transparent overlay for highlighting screen regions.

    Creates a borderless, transparent window that displays a highlight shape
    (box, circle, or arrow) at a specified screen location. The overlay is
    non-blocking and automatically dismisses after the configured duration.

    Supports:
    - Multiple shapes: box, circle, arrow
    - Configurable color, thickness, and duration
    - Thread-safe operation
    - Automatic cleanup
    """

    def __init__(
        self,
        x: int,
        y: int,
        duration_ms: int = 2000,
        color: str = "#FF0000",
        thickness: int = 3,
        style: Literal["box", "circle", "arrow"] = "box",
        size: int = 100,
    ):
        """Initialize highlight overlay.

        Args:
            x: X coordinate of highlight center
            y: Y coordinate of highlight center
            duration_ms: Display duration in milliseconds
            color: Hex color code (e.g., "#FF0000")
            thickness: Border thickness in pixels
            style: Highlight style - "box", "circle", or "arrow"
            size: Size of the highlight shape in pixels
        """
        self.x = x
        self.y = y
        self.duration_ms = duration_ms
        self.color = color
        self.thickness = thickness
        self.style = style
        self.size = size
        self.root: tk.Tk | None = None
        self.canvas: tk.Canvas | None = None

    def show(self) -> None:
        """Display the highlight overlay in a separate thread.

        This method is non-blocking and returns immediately after starting
        the overlay thread.
        """
        thread = threading.Thread(target=self._create_and_run, daemon=True)
        thread.start()

    def _create_and_run(self) -> None:
        """Create and run the overlay window (runs in separate thread)."""
        try:
            # Create root window
            self.root = tk.Tk()

            # Configure window to be transparent and always on top
            self.root.attributes("-topmost", True)
            self.root.attributes("-alpha", 0.7)  # 70% opacity
            self.root.overrideredirect(True)  # Remove window decorations

            # Try to make window click-through (platform-specific)
            try:
                # Works on some platforms
                self.root.wm_attributes("-transparentcolor", "white")
            except tk.TclError:
                # Not supported on all platforms, continue anyway
                pass

            # Set window size and position
            window_size = self.size + 40  # Add padding
            self.root.geometry(
                f"{window_size}x{window_size}+{self.x - window_size // 2}+{self.y - window_size // 2}"
            )

            # Create canvas for drawing
            self.canvas = tk.Canvas(
                self.root,
                width=window_size,
                height=window_size,
                bg="white",
                highlightthickness=0,
            )
            self.canvas.pack()

            # Draw the highlight shape
            self._draw_shape()

            # Schedule window destruction after duration
            self.root.after(self.duration_ms, self._destroy)

            # Run the event loop
            self.root.mainloop()

        except Exception as e:
            logger.error(f"Error creating highlight overlay: {e}")
            if self.root:
                try:
                    self.root.destroy()
                except Exception:
                    pass

    def _draw_shape(self) -> None:
        """Draw the highlight shape on the canvas."""
        if not self.canvas:
            return

        center = self.size // 2 + 20  # Center point with padding
        half_size = self.size // 2

        if self.style == "box":
            self._draw_box(center, half_size)
        elif self.style == "circle":
            self._draw_circle(center, half_size)
        elif self.style == "arrow":
            self._draw_arrow(center, half_size)
        else:
            logger.warning(f"Unknown highlight style: {self.style}, using box")
            self._draw_box(center, half_size)

    def _draw_box(self, center: int, half_size: int) -> None:
        """Draw a rectangular box.

        Args:
            center: Center coordinate
            half_size: Half the size of the box
        """
        if not self.canvas:
            return

        x1 = center - half_size
        y1 = center - half_size
        x2 = center + half_size
        y2 = center + half_size

        self.canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            outline=self.color,
            width=self.thickness,
        )

    def _draw_circle(self, center: int, half_size: int) -> None:
        """Draw a circle.

        Args:
            center: Center coordinate
            half_size: Radius of the circle
        """
        if not self.canvas:
            return

        x1 = center - half_size
        y1 = center - half_size
        x2 = center + half_size
        y2 = center + half_size

        self.canvas.create_oval(
            x1,
            y1,
            x2,
            y2,
            outline=self.color,
            width=self.thickness,
        )

    def _draw_arrow(self, center: int, half_size: int) -> None:
        """Draw a downward-pointing arrow.

        Args:
            center: Center coordinate
            half_size: Half the size of the arrow
        """
        if not self.canvas:
            return

        # Arrow shaft (vertical line)
        shaft_top_y = center - half_size
        shaft_bottom_y = center + half_size // 2

        self.canvas.create_line(
            center,
            shaft_top_y,
            center,
            shaft_bottom_y,
            fill=self.color,
            width=self.thickness,
        )

        # Arrow head (triangle pointing down)
        arrow_width = half_size // 2

        # Triangle points
        points = [
            center - arrow_width,
            shaft_bottom_y,  # Left point
            center + arrow_width,
            shaft_bottom_y,  # Right point
            center,
            center + half_size,  # Bottom point
        ]

        self.canvas.create_polygon(
            points,
            fill=self.color,
            outline=self.color,
            width=self.thickness,
        )

    def _destroy(self) -> None:
        """Destroy the overlay window."""
        if self.root:
            try:
                self.root.destroy()
            except Exception as e:
                logger.debug(f"Error destroying overlay: {e}")
