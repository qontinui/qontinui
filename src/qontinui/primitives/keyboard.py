"""Keyboard primitives - ported from Qontinui framework.

Each primitive keyboard action does exactly one thing and extends Action
to get lifecycle management capabilities.
"""

from ..actions import Action, ActionConfig, TypeOptions
from ..actions.action_options import KeyDownOptions as KeyDownOptionsLegacy
from ..actions.basic.type.key_up_options import KeyUpOptions, KeyUpOptionsBuilder
from ..actions.pure import ActionResult, PureActions


class KeyPress(Action):
    """Primitive key press action.

    Port of KeyPress from Qontinui framework primitive.
    Presses and releases a single key.
    """

    def __init__(self, config: ActionConfig | None = None) -> None:
        """Initialize with optional ActionConfig.

        Args:
            config: ActionConfig instance or None for defaults
        """
        super().__init__()
        self._config = config or ActionConfig()
        self._pure = PureActions()

    def execute_key(self, key: str) -> ActionResult:
        """Execute key press.

        Args:
            key: Key to press

        Returns:
            ActionResult
        """
        return self.execute(lambda: self._pure.key_press(key), target=key)


class KeyDown(Action):
    """Primitive key down action.

    Port of KeyDown from Qontinui framework primitive.
    Presses and holds a key.
    """

    def __init__(self, config: KeyDownOptionsLegacy | None = None) -> None:
        """Initialize with optional KeyDownOptions.

        Args:
            config: KeyDownOptions instance or None for defaults
        """
        super().__init__()
        self._config = config or KeyDownOptionsLegacy()
        self._pure = PureActions()

    def execute_key(self, key: str) -> ActionResult:
        """Execute key down.

        Args:
            key: Key to press down

        Returns:
            ActionResult
        """
        hold_duration = getattr(self._config, "_hold_duration", 0.0)
        modifiers = getattr(self._config, "_modifiers", [])

        def key_down_action():
            # Press modifiers first
            for modifier in modifiers:
                mod_result = self._pure.key_down(modifier.value)
                if not mod_result.success:
                    return mod_result

            # Press the main key
            result = self._pure.key_down(key)

            # Hold if duration specified
            if hold_duration > 0 and result.success:
                self._pure.wait(hold_duration)
                # Auto-release after hold
                self._pure.key_up(key)
                # Release modifiers
                for modifier in reversed(modifiers):
                    self._pure.key_up(modifier.value)

            return result

        return self.execute(key_down_action, target=key)


class KeyUp(Action):
    """Primitive key up action.

    Port of KeyUp from Qontinui framework primitive.
    Releases a held key.
    """

    def __init__(self, config: KeyUpOptions | None = None) -> None:
        """Initialize with optional KeyUpOptions.

        Args:
            config: KeyUpOptions instance or None for defaults
        """
        super().__init__()
        self._config = config or KeyUpOptionsBuilder().build()
        self._pure = PureActions()

    def execute_key(self, key: str) -> ActionResult:
        """Execute key up.

        Args:
            key: Key to release

        Returns:
            ActionResult
        """
        release_modifiers = getattr(self._config, "_release_modifiers", True)

        def key_up_action():
            # Release the main key
            result = self._pure.key_up(key)

            # Release modifiers if configured
            if release_modifiers:
                # Common modifiers to check and release
                for mod in ["ctrl", "alt", "shift", "meta", "cmd", "win"]:
                    try:
                        self._pure.key_up(mod)
                    except (OSError, RuntimeError, ValueError):
                        # OK to ignore if modifier wasn't pressed
                        pass

            return result

        return self.execute(key_up_action, target=key)


class TypeText(Action):
    """Primitive text typing action.

    Port of TypeText from Qontinui framework primitive.
    Types a string of text character by character.
    """

    def __init__(self, config: TypeOptions | None = None) -> None:
        """Initialize with optional TypeOptions.

        Args:
            config: TypeOptions instance or None for defaults
        """
        super().__init__()
        if config is None:
            from ..actions.basic.type import TypeOptionsBuilder

            config = TypeOptionsBuilder().build()
        self._config = config
        self._pure = PureActions()

    def execute_text(self, text: str) -> ActionResult:
        """Execute text typing.

        Args:
            text: Text to type

        Returns:
            ActionResult
        """
        typing_delay = getattr(self._config, "_typing_delay", 0.05)
        clear_before = getattr(self._config, "_clear_before", False)
        press_enter = getattr(self._config, "_press_enter", False)
        modifiers = getattr(self._config, "_modifiers", [])

        def type_action():
            # Clear field if configured
            if clear_before:
                # Select all (Ctrl+A) and delete
                self._pure.key_down("ctrl")
                self._pure.key_press("a")
                self._pure.key_up("ctrl")
                self._pure.key_press("delete")

            # Press modifiers if any
            for modifier in modifiers:
                mod_result = self._pure.key_down(modifier.value)
                if not mod_result.success:
                    return mod_result

            # Type each character
            for char in text:
                result = self._pure.type_character(char)
                if not result.success:
                    # Release modifiers on failure
                    for modifier in reversed(modifiers):
                        self._pure.key_up(modifier.value)
                    return result

                # Delay between characters
                if typing_delay > 0 and char != text[-1]:  # No delay after last char
                    self._pure.wait(typing_delay)

            # Release modifiers
            for modifier in reversed(modifiers):
                self._pure.key_up(modifier.value)

            # Press enter if configured
            if press_enter:
                self._pure.key_press("enter")

            return ActionResult(success=True, data={"text": text, "length": len(text)})

        return self.execute(type_action, target=text)
