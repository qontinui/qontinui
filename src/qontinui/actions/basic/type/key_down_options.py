"""Key down options - ported from Qontinui framework.

Configuration for keyboard key press-and-hold operations.
"""

from ...action_config import ActionConfig, ActionConfigBuilder


class KeyDownOptions(ActionConfig):
    """Configuration for KeyDown actions.

    Port of KeyDownOptions from Qontinui framework class.

    Configures keyboard key press operations where keys are
    held down without releasing.

    This class is immutable and must be constructed using its Builder.

    Example usage:
        key_down_opts = KeyDownOptionsBuilder()
            .add_key("a")
            .add_keys("b", "c")
            .with_modifiers("ctrl", "shift")
            .with_pause(0.2)
            .build()
    """

    def __init__(self, builder: "KeyDownOptionsBuilder") -> None:
        """Initialize KeyDownOptions from builder.

        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self.keys: tuple[str, ...] = builder.keys
        self.modifiers: tuple[str, ...] = builder.modifiers
        self.pause_between_keys: float = builder.pause_between_keys

    def get_keys(self) -> tuple[str, ...]:
        """Get the keys to press.

        Returns:
            Tuple of key strings
        """
        return self.keys

    def get_modifiers(self) -> tuple[str, ...]:
        """Get the modifier keys.

        Returns:
            Tuple of modifier key strings
        """
        return self.modifiers

    def get_pause_between_keys(self) -> float:
        """Get the pause duration between pressing multiple keys.

        Returns:
            Pause duration in seconds
        """
        return self.pause_between_keys


class KeyDownOptionsBuilder(ActionConfigBuilder):
    """Builder for constructing KeyDownOptions with a fluent API.

    Port of KeyDownOptions from Qontinui framework.Builder.
    """

    def __init__(self, original: KeyDownOptions | None = None) -> None:
        """Initialize builder.

        Args:
            original: Optional KeyDownOptions instance to copy values from
        """
        super().__init__(original)

        if original:
            self.keys = original.keys
            self.modifiers = original.modifiers
            self.pause_between_keys = original.pause_between_keys
        else:
            self.keys = ()
            self.modifiers = ()
            self.pause_between_keys = 0.1

    def add_key(self, key: str) -> "KeyDownOptionsBuilder":
        """Add a key to press.

        Args:
            key: Key to add

        Returns:
            This builder instance for chaining
        """
        self.keys = self.keys + (key,)
        return self

    def add_keys(self, *keys: str) -> "KeyDownOptionsBuilder":
        """Add multiple keys to press.

        Args:
            *keys: Keys to add

        Returns:
            This builder instance for chaining
        """
        self.keys = self.keys + keys
        return self

    def with_modifiers(self, *modifiers: str) -> "KeyDownOptionsBuilder":
        """Set modifier keys.

        Args:
            *modifiers: Modifier keys (ctrl, shift, alt, cmd)

        Returns:
            This builder instance for chaining
        """
        self.modifiers = modifiers
        return self

    def with_pause(self, seconds: float) -> "KeyDownOptionsBuilder":
        """Set pause between keys.

        Args:
            seconds: Pause duration

        Returns:
            This builder instance for chaining
        """
        self.pause_between_keys = seconds
        return self

    def build(self) -> KeyDownOptions:
        """Build the immutable KeyDownOptions object.

        Returns:
            A new instance of KeyDownOptions
        """
        return KeyDownOptions(self)

    def _self(self) -> "KeyDownOptionsBuilder":
        """Return self for fluent interface.

        Returns:
            This builder instance
        """
        return self
