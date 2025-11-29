"""Type action - ported from Qontinui framework.

Text typing action with various input methods.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, cast

from ...action_config import ActionConfig
from ...action_interface import ActionInterface
from ...action_result import ActionResult
from ...action_type import ActionType
from ...object_collection import ObjectCollection


class TypeMethod(Enum):
    """Methods for typing text."""

    KEYBOARD = auto()  # Regular keyboard input
    PASTE = auto()  # Paste from clipboard
    FIELD = auto()  # Type into specific field
    SECURE = auto()  # Secure typing (no logging)


@dataclass
class TypeOptions(ActionConfig):
    """Options for type actions.

    Port of TypeOptions from Qontinui framework class.
    """

    type_method: TypeMethod = TypeMethod.KEYBOARD
    clear_before: bool = False  # Clear field before typing
    select_all: bool = False  # Select all before typing
    type_delay: float = 0.0  # Delay between characters
    verify_text: bool = False  # Verify text after typing
    secure_mode: bool = False  # Don't log typed text

    def keyboard(self) -> "TypeOptions":
        """Use keyboard typing.

        Returns:
            Self for fluent interface
        """
        self.type_method = TypeMethod.KEYBOARD
        return self

    def paste(self) -> "TypeOptions":
        """Use paste method.

        Returns:
            Self for fluent interface
        """
        self.type_method = TypeMethod.PASTE
        return self

    def clear_first(self) -> "TypeOptions":
        """Clear field before typing.

        Returns:
            Self for fluent interface
        """
        self.clear_before = True
        return self

    def select_all_first(self) -> "TypeOptions":
        """Select all before typing.

        Returns:
            Self for fluent interface
        """
        self.select_all = True
        return self

    def with_delay(self, delay: float) -> "TypeOptions":
        """Set delay between characters.

        Args:
            delay: Delay in seconds

        Returns:
            Self for fluent interface
        """
        self.type_delay = delay
        return self

    def verify(self) -> "TypeOptions":
        """Enable text verification after typing.

        Returns:
            Self for fluent interface
        """
        self.verify_text = True
        return self

    def secure(self) -> "TypeOptions":
        """Enable secure mode (no logging).

        Returns:
            Self for fluent interface
        """
        self.secure_mode = True
        self.type_method = TypeMethod.SECURE
        return self


class TypeAction(ActionInterface):
    """Type action implementation.

    Port of Type from Qontinui framework class.
    Named TypeAction to avoid conflict with Python's type keyword.

    Provides text input functionality with various methods
    and options for clearing, selecting, and verifying.
    """

    def __init__(self, options: TypeOptions | None = None) -> None:
        """Initialize Type action.

        Args:
            options: Type options
        """
        self.options = options or TypeOptions()
        self._last_typed_text: str | None = None

    def get_action_type(self) -> ActionType:
        """Return the action type.

        Returns:
            ActionType.TYPE
        """
        return ActionType.TYPE

    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Execute the type action using the Qontinui framework pattern.

        Args:
            matches: Contains ActionOptions and accumulates execution results
            object_collections: Collections containing text to type (as Strings)
        """
        # Extract text from object collections
        text_to_type = ""
        for collection in object_collections:
            if collection.state_strings:
                text_to_type = collection.state_strings[0].get_string()  # Use first string
                break

        # Execute the type action
        success = self.execute(text_to_type)

        # Update matches with results
        object.__setattr__(matches, "success", success)
        if success:
            matches.add_text_result(text_to_type)  # type: ignore[attr-defined]

    def execute(self, text: str, target: Any | None = None) -> bool:
        """Execute type action.

        Args:
            text: Text to type
            target: Optional target field/region

        Returns:
            True if typing successful
        """
        # Focus target if provided
        if target:
            if not self._focus_target(target):
                return False

        # Apply pre-action pause
        self._pause_before()

        # Clear or select if requested
        if self.options.clear_before:
            self._clear_field()
        elif self.options.select_all:
            self._select_all()

        # Type the text
        success = False
        if self.options.type_method == TypeMethod.PASTE:
            success = self._paste_text(text)
        elif self.options.type_method == TypeMethod.SECURE:
            success = self._type_secure(text)
        else:
            success = self._type_text(text)

        if success:
            if not self.options.secure_mode:
                self._last_typed_text = text

                # Emit TEXT_TYPED event for runner/frontend
                from ....reporting.events import EventType, emit_event

                emit_event(EventType.TEXT_TYPED, {"text": text, "character_count": len(text)})

            # Verify if requested
            if self.options.verify_text:
                success = self._verify_text(text)

        # Apply post-action pause
        self._pause_after()

        return success

    def _focus_target(self, target: Any) -> bool:
        """Focus on target field/region.

        Args:
            target: Target to focus

        Returns:
            True if focused successfully
        """
        # Click on target to focus
        from ..click.click import Click

        return cast(bool, Click().execute(target))

    def _clear_field(self):
        """Clear the current field."""
        self._select_all()
        self._key_press("delete")

    def _select_all(self):
        """Select all text in current field."""
        self._key_combo("ctrl", "a")

    def _type_text(self, text: str) -> bool:
        """Type text using keyboard.

        Args:
            text: Text to type

        Returns:
            True if successful
        """
        try:
            for char in text:
                self._type_char(char)
                if self.options.type_delay > 0:
                    self._pause(self.options.type_delay)
            return True
        except Exception as e:
            print(f"Type error: {e}")
            return False

    def _paste_text(self, text: str) -> bool:
        """Paste text from clipboard.

        Args:
            text: Text to paste

        Returns:
            True if successful
        """
        # Set clipboard
        self._set_clipboard(text)
        # Paste
        self._key_combo("ctrl", "v")
        return True

    def _type_secure(self, text: str) -> bool:
        """Type text securely (no logging).

        Args:
            text: Text to type

        Returns:
            True if successful
        """
        # Type without logging
        return self._type_text(text)

    def _verify_text(self, expected: str) -> bool:
        """Verify typed text matches expected.

        Args:
            expected: Expected text

        Returns:
            True if matches
        """
        # This would use OCR or field reading
        # For now, assume success
        return True

    def _type_char(self, char: str):
        """Type a single character.

        Args:
            char: Character to type
        """
        if not self.options.secure_mode:
            print(f"Type: {char}")

    def _key_press(self, key: str):
        """Press a key.

        Args:
            key: Key to press
        """
        print(f"Key press: {key}")

    def _key_combo(self, *keys: str):
        """Press key combination.

        Args:
            *keys: Keys to press together
        """
        print(f"Key combo: {'+'.join(keys)}")

    def _set_clipboard(self, text: str):
        """Set clipboard content.

        Args:
            text: Text to set
        """
        print(f"Set clipboard: {'[SECURE]' if self.options.secure_mode else text}")

    def _pause_before(self):
        """Apply pre-action pause from options."""
        if self.options.pause_before > 0:
            self._pause(self.options.pause_before)

    def _pause_after(self):
        """Apply post-action pause from options."""
        if self.options.pause_after > 0:
            self._pause(self.options.pause_after)

    def _pause(self, duration: float):
        """Pause for specified duration.

        Args:
            duration: Duration in seconds
        """
        import time

        time.sleep(duration)

    def get_last_typed(self) -> str | None:
        """Get last typed text.

        Returns:
            Last typed text or None (None if secure mode)
        """
        return self._last_typed_text

    @staticmethod
    def type_text(text: str) -> bool:
        """Convenience method for typing text.

        Args:
            text: Text to type

        Returns:
            True if successful
        """
        return TypeAction().execute(text)

    @staticmethod
    def paste_text(text: str) -> bool:
        """Convenience method for pasting text.

        Args:
            text: Text to paste

        Returns:
            True if successful
        """
        return TypeAction(TypeOptions().paste()).execute(text)
