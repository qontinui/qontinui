"""MouseButton enum - ported from Qontinui framework.

Represents the physical mouse buttons.
"""

from enum import Enum


class MouseButton(Enum):
    """Represents the physical mouse buttons.

    Port of MouseButton from Qontinui framework enum.

    This enum has a single responsibility: identifying which mouse button
    is being used. It does not concern itself with click types (single/double)
    or timing behaviors, following the Single Responsibility Principle.
    """

    LEFT = "LEFT"
    """The primary (left) mouse button.
    Typically used for selection and primary actions.
    """

    RIGHT = "RIGHT"
    """The secondary (right) mouse button.
    Typically used for context menus and secondary actions.
    """

    MIDDLE = "MIDDLE"
    """The middle mouse button or scroll wheel click.
    Often used for auxiliary functions like opening links in new tabs.
    """

    @classmethod
    def from_string(cls, button: str) -> "MouseButton":
        """Get MouseButton from string.

        Args:
            button: Button name (case-insensitive)

        Returns:
            MouseButton enum value

        Raises:
            ValueError: If button name is invalid
        """
        try:
            return cls[button.upper()]
        except KeyError as e:
            raise ValueError(f"Invalid mouse button: {button}") from e

    def to_pyautogui(self) -> str:
        """Convert to PyAutoGUI button string.

        Returns:
            Button string for PyAutoGUI
        """
        return self.value.lower()

    def is_primary(self) -> bool:
        """Check if this is the primary (left) button.

        Returns:
            True if LEFT button
        """
        return self == MouseButton.LEFT

    def is_secondary(self) -> bool:
        """Check if this is the secondary (right) button.

        Returns:
            True if RIGHT button
        """
        return self == MouseButton.RIGHT

    def is_auxiliary(self) -> bool:
        """Check if this is an auxiliary (middle) button.

        Returns:
            True if MIDDLE button
        """
        return self == MouseButton.MIDDLE
