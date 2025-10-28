"""MockKeyboard - Simulates keyboard operations (Brobot pattern).

Provides mock keyboard operations for testing without actual keyboard control.
Tracks all keyboard operations and provides query capabilities, but doesn't
actually send keystrokes.

This enables:
- Fast testing (no real keyboard operations)
- Headless CI/CD execution (no display needed)
- Deterministic testing (no actual GUI interaction)
- Operation tracking (verify correct operations were performed)
"""

import logging

logger = logging.getLogger(__name__)


class MockKeyboard:
    """Mock keyboard implementation.

    Simulates keyboard operations by tracking them without actually
    performing keyboard control. Useful for testing automation logic
    without requiring a GUI environment.

    Example:
        keyboard = MockKeyboard()
        keyboard.type_text("Hello")  # Logs but doesn't type
        keyboard.press("enter")  # Logs but doesn't press
        keyboard.hotkey("ctrl", "c")  # Logs but doesn't send hotkey
    """

    def __init__(self) -> None:
        """Initialize MockKeyboard."""
        # Track operation history
        self.operations: list[dict] = []

        # Track currently pressed keys
        self.pressed_keys: set[str] = set()

        logger.debug("MockKeyboard initialized")

    def type_text(self, text: str, interval: float = 0.0) -> bool:
        """Type text string (mock).

        Args:
            text: Text to type
            interval: Interval between characters (ignored)

        Returns:
            Always True
        """
        logger.debug(f"MockKeyboard.type_text: '{text[:50]}...', interval={interval}")

        self.operations.append(
            {
                "type": "type_text",
                "text": text,
                "interval": interval,
            }
        )

        return True

    def press(self, key: str, presses: int = 1) -> bool:
        """Press key (mock).

        Args:
            key: Key to press
            presses: Number of presses

        Returns:
            Always True
        """
        logger.debug(f"MockKeyboard.press: {key} x{presses}")

        self.operations.append(
            {
                "type": "press",
                "key": key,
                "presses": presses,
            }
        )

        return True

    def hotkey(self, *keys: str) -> bool:
        """Press key combination (mock).

        Args:
            *keys: Keys to press together

        Returns:
            Always True
        """
        logger.debug(f"MockKeyboard.hotkey: {'+'.join(keys)}")

        self.operations.append(
            {
                "type": "hotkey",
                "keys": list(keys),
            }
        )

        return True

    def key_down(self, key: str) -> bool:
        """Press and hold key (mock).

        Args:
            key: Key to press down

        Returns:
            Always True
        """
        logger.debug(f"MockKeyboard.key_down: {key}")

        self.pressed_keys.add(key)

        self.operations.append(
            {
                "type": "key_down",
                "key": key,
            }
        )

        return True

    def key_up(self, key: str) -> bool:
        """Release key (mock).

        Args:
            key: Key to release

        Returns:
            Always True
        """
        logger.debug(f"MockKeyboard.key_up: {key}")

        self.pressed_keys.discard(key)

        self.operations.append(
            {
                "type": "key_up",
                "key": key,
            }
        )

        return True

    def is_key_pressed(self, key: str) -> bool:
        """Check if key is pressed (mock).

        Args:
            key: Key to check

        Returns:
            True if key is in pressed set
        """
        return key in self.pressed_keys

    # Utility methods for testing

    def reset(self) -> None:
        """Reset operation history and pressed keys.

        Example:
            keyboard = MockKeyboard()
            keyboard.type_text("Hello")
            keyboard.reset()  # Clear history
        """
        self.operations.clear()
        self.pressed_keys.clear()
        logger.debug("MockKeyboard reset")

    def get_last_operation(self) -> dict | None:
        """Get last operation performed.

        Returns:
            Last operation dict or None

        Example:
            keyboard = MockKeyboard()
            keyboard.type_text("Hello")
            op = keyboard.get_last_operation()
            assert op["type"] == "type_text"
            assert op["text"] == "Hello"
        """
        return self.operations[-1] if self.operations else None

    def get_operation_count(self, operation_type: str | None = None) -> int:
        """Get count of operations.

        Args:
            operation_type: Filter by type (None = all)

        Returns:
            Count of operations

        Example:
            keyboard = MockKeyboard()
            keyboard.type_text("Hello")
            keyboard.press("enter")
            assert keyboard.get_operation_count("type_text") == 1
            assert keyboard.get_operation_count() == 2
        """
        if operation_type is None:
            return len(self.operations)

        return sum(1 for op in self.operations if op["type"] == operation_type)

    def get_typed_text(self) -> str:
        """Get all text that was typed.

        Returns:
            Concatenated text from all type_text operations

        Example:
            keyboard = MockKeyboard()
            keyboard.type_text("Hello ")
            keyboard.type_text("World")
            assert keyboard.get_typed_text() == "Hello World"
        """
        text_operations = [op for op in self.operations if op["type"] == "type_text"]
        return "".join(op["text"] for op in text_operations)
