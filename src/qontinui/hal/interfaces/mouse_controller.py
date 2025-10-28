"""Mouse controller interface definition."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class MouseButton(Enum):
    """Mouse button enumeration."""

    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


@dataclass
class MousePosition:
    """Current mouse position."""

    x: int
    y: int


class IMouseController(ABC):
    """Interface for mouse control operations.

    This interface defines mouse-only operations following the
    Single Responsibility Principle. Implementations should focus
    solely on mouse input control.
    """

    @abstractmethod
    def mouse_move(self, x: int, y: int, duration: float = 0.0) -> bool:
        """Move mouse to absolute position.

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Movement duration in seconds (0 for instant)

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def mouse_move_relative(self, dx: int, dy: int, duration: float = 0.0) -> bool:
        """Move mouse relative to current position.

        Args:
            dx: X offset
            dy: Y offset
            duration: Movement duration in seconds

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def mouse_click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: MouseButton = MouseButton.LEFT,
        clicks: int = 1,
        interval: float = 0.0,
    ) -> bool:
        """Click mouse button.

        Args:
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)
            button: Mouse button to click
            clicks: Number of clicks
            interval: Interval between clicks in seconds

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def mouse_down(
        self, x: int | None = None, y: int | None = None, button: MouseButton = MouseButton.LEFT
    ) -> bool:
        """Press and hold mouse button.

        Args:
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)
            button: Mouse button to press

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def mouse_up(
        self, x: int | None = None, y: int | None = None, button: MouseButton = MouseButton.LEFT
    ) -> bool:
        """Release mouse button.

        Args:
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)
            button: Mouse button to release

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def mouse_drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        button: MouseButton = MouseButton.LEFT,
        duration: float = 0.5,
    ) -> bool:
        """Drag mouse from start to end position.

        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            button: Mouse button to hold during drag
            duration: Drag duration in seconds

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def mouse_scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> bool:
        """Scroll mouse wheel.

        Args:
            clicks: Number of scroll clicks (positive=up, negative=down)
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def get_mouse_position(self) -> MousePosition:
        """Get current mouse position.

        Returns:
            Current mouse position
        """
        pass

    @abstractmethod
    def click_at(self, x: int, y: int, button: MouseButton = MouseButton.LEFT) -> bool:
        """Click at specific coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button to click

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def double_click_at(self, x: int, y: int, button: MouseButton = MouseButton.LEFT) -> bool:
        """Double click at specific coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button to click

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def drag(
        self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.0
    ) -> bool:
        """Drag from start to end position.

        Args:
            start_x: Start X coordinate
            start_y: Start Y coordinate
            end_x: End X coordinate
            end_y: End Y coordinate
            duration: Drag duration in seconds

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def move_mouse(self, x: int, y: int, duration: float = 0.0) -> bool:
        """Move mouse to position (alias for mouse_move).

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Movement duration in seconds

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> bool:
        """Scroll mouse wheel (alias for mouse_scroll).

        Args:
            clicks: Number of scroll clicks (positive=up, negative=down)
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)

        Returns:
            True if successful
        """
        pass
