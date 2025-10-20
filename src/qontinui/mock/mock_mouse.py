"""MockMouse - Simulates mouse operations (Brobot pattern).

Provides mock mouse operations for testing without actual mouse control.
Tracks all mouse operations and provides query capabilities, but doesn't
actually move or click the mouse.

This enables:
- Fast testing (no real mouse operations)
- Headless CI/CD execution (no display needed)
- Deterministic testing (no actual GUI interaction)
- Operation tracking (verify correct operations were performed)
"""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..hal.interfaces.input_controller import MouseButton, MousePosition

logger = logging.getLogger(__name__)


class MockMouse:
    """Mock mouse implementation.

    Simulates mouse operations by tracking them without actually
    performing mouse control. Useful for testing automation logic
    without requiring a GUI environment.

    Example:
        mouse = MockMouse()
        mouse.move(100, 200)  # Logs but doesn't move mouse
        mouse.click()  # Logs but doesn't click
        pos = mouse.get_position()  # Returns tracked position
    """

    def __init__(self):
        """Initialize MockMouse."""
        # Track virtual mouse position
        self._x = 0
        self._y = 0

        # Track operation history
        self.operations: list[dict] = []

        logger.debug("MockMouse initialized")

    def move(self, x: int, y: int, duration: float = 0.0) -> bool:
        """Move mouse to position (mock).

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Movement duration (ignored in mock)

        Returns:
            Always True
        """
        logger.debug(f"MockMouse.move: ({x}, {y}), duration={duration}")

        self._x = x
        self._y = y

        self.operations.append(
            {
                "type": "move",
                "x": x,
                "y": y,
                "duration": duration,
            }
        )

        return True

    def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: Optional["MouseButton"] = None,
        clicks: int = 1,
    ) -> bool:
        """Click mouse button (mock).

        Args:
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)
            button: Mouse button (ignored in mock)
            clicks: Number of clicks

        Returns:
            Always True
        """
        # Update position if specified
        if x is not None:
            self._x = x
        if y is not None:
            self._y = y

        button_name = button.value if button else "left"
        logger.debug(
            f"MockMouse.click: ({self._x}, {self._y}), button={button_name}, clicks={clicks}"
        )

        self.operations.append(
            {
                "type": "click",
                "x": self._x,
                "y": self._y,
                "button": button_name,
                "clicks": clicks,
            }
        )

        return True

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = 0.5,
    ) -> bool:
        """Drag mouse (mock).

        Args:
            start_x: Start X coordinate
            start_y: Start Y coordinate
            end_x: End X coordinate
            end_y: End Y coordinate
            duration: Drag duration (ignored)

        Returns:
            Always True
        """
        logger.debug(f"MockMouse.drag: ({start_x},{start_y}) → ({end_x},{end_y})")

        # Update position to end location
        self._x = end_x
        self._y = end_y

        self.operations.append(
            {
                "type": "drag",
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "duration": duration,
            }
        )

        return True

    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> bool:
        """Scroll mouse wheel (mock).

        Args:
            clicks: Number of clicks (positive=up, negative=down)
            x: X coordinate (optional)
            y: Y coordinate (optional)

        Returns:
            Always True
        """
        logger.debug(f"MockMouse.scroll: clicks={clicks}, pos=({x}, {y})")

        self.operations.append(
            {
                "type": "scroll",
                "clicks": clicks,
                "x": x,
                "y": y,
            }
        )

        return True

    def get_position(self) -> "MousePosition":
        """Get current mouse position (mock).

        Returns:
            Tracked mouse position
        """
        from ..hal.interfaces.input_controller import MousePosition

        logger.debug(f"MockMouse.get_position: ({self._x}, {self._y})")
        return MousePosition(x=self._x, y=self._y)

    # Utility methods for testing

    def reset(self) -> None:
        """Reset tracked position and operation history.

        Example:
            mouse = MockMouse()
            mouse.click(100, 200)
            mouse.reset()  # Clear history
        """
        self._x = 0
        self._y = 0
        self.operations.clear()
        logger.debug("MockMouse reset")

    def get_last_operation(self) -> dict | None:
        """Get last operation performed.

        Returns:
            Last operation dict or None

        Example:
            mouse = MockMouse()
            mouse.click(100, 200)
            op = mouse.get_last_operation()
            assert op["type"] == "click"
        """
        return self.operations[-1] if self.operations else None

    def get_operation_count(self, operation_type: str | None = None) -> int:
        """Get count of operations.

        Args:
            operation_type: Filter by type (None = all)

        Returns:
            Count of operations

        Example:
            mouse = MockMouse()
            mouse.click(100, 200)
            mouse.click(300, 400)
            assert mouse.get_operation_count("click") == 2
        """
        if operation_type is None:
            return len(self.operations)

        return sum(1 for op in self.operations if op["type"] == operation_type)
