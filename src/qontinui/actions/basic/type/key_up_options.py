"""Key up options - ported from Qontinui framework.

Configuration for keyboard key release operations.
"""

from ...action_config import ActionConfig, ActionConfigBuilder


class KeyUpOptions(ActionConfig):
    """Configuration for KeyUp actions.

    Port of KeyUpOptions from Qontinui framework class.

    Configures keyboard key release operations to complete
    key press sequences started with KeyDown.

    This is an immutable object and must be constructed using its Builder.

    Example usage:
        options = KeyUpOptionsBuilder()
            .add_key("a")
            .add_keys("b", "c")
            .with_modifiers("ctrl", "shift")
            .set_pause_between_keys(0.15)
            .set_release_modifiers_first(True)
            .build()
    """

    def __init__(self, builder: "KeyUpOptionsBuilder") -> None:
        """Initialize KeyUpOptions from builder.

        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self._keys: tuple[str, ...] = builder.keys
        self._modifiers: tuple[str, ...] = builder.modifiers
        self._pause_between_keys: float = builder.pause_between_keys
        self._release_modifiers_first: bool = builder.release_modifiers_first

    def get_keys(self) -> tuple[str, ...]:
        """Get keys to release.

        Returns:
            Tuple of key names
        """
        return self._keys

    def get_modifiers(self) -> tuple[str, ...]:
        """Get modifier keys to release.

        Returns:
            Tuple of modifier key names
        """
        return self._modifiers

    def get_pause_between_keys(self) -> float:
        """Get pause duration between releasing multiple keys.

        Returns:
            Pause duration in seconds
        """
        return self._pause_between_keys

    def get_release_modifiers_first(self) -> bool:
        """Get whether to release modifiers before other keys.

        Returns:
            True if modifiers should be released first
        """
        return self._release_modifiers_first

    # Provide property aliases for backward compatibility with direct field access
    @property
    def keys(self) -> tuple[str, ...]:
        """Get keys to release (property access)."""
        return self._keys

    @property
    def modifiers(self) -> tuple[str, ...]:
        """Get modifier keys to release (property access)."""
        return self._modifiers

    @property
    def pause_between_keys(self) -> float:
        """Get pause duration between releasing multiple keys (property access)."""
        return self._pause_between_keys

    @property
    def release_modifiers_first(self) -> bool:
        """Get whether to release modifiers before other keys (property access)."""
        return self._release_modifiers_first


class KeyUpOptionsBuilder(ActionConfigBuilder):
    """Builder for constructing KeyUpOptions with a fluent API.

    Port of KeyUpOptions from Qontinui framework.Builder.
    """

    def __init__(self, original: KeyUpOptions | None = None) -> None:
        """Initialize builder.

        Args:
            original: Optional KeyUpOptions instance to copy values from
        """
        super().__init__(original)

        # Initialize attributes with proper types
        self.keys: tuple[str, ...]
        self.modifiers: tuple[str, ...]
        self.pause_between_keys: float
        self.release_modifiers_first: bool

        if original:
            self.keys = original.get_keys()
            self.modifiers = original.get_modifiers()
            self.pause_between_keys = original.get_pause_between_keys()
            self.release_modifiers_first = original.get_release_modifiers_first()
        else:
            self.keys = ()
            self.modifiers = ()
            self.pause_between_keys = 0.1
            self.release_modifiers_first = False

    def add_key(self, key: str) -> "KeyUpOptionsBuilder":
        """Add a key to release.

        Args:
            key: Key to add

        Returns:
            This builder instance for chaining
        """
        self.keys = self.keys + (key,)
        return self

    def add_keys(self, *keys: str) -> "KeyUpOptionsBuilder":
        """Add multiple keys to release.

        Args:
            *keys: Keys to add

        Returns:
            This builder instance for chaining
        """
        self.keys = self.keys + keys
        return self

    def with_modifiers(self, *modifiers: str) -> "KeyUpOptionsBuilder":
        """Set modifier keys to release.

        Args:
            *modifiers: Modifier keys (ctrl, shift, alt, cmd)

        Returns:
            This builder instance for chaining
        """
        self.modifiers = modifiers
        return self

    def set_pause_between_keys(self, seconds: float) -> "KeyUpOptionsBuilder":
        """Set pause between key releases.

        Args:
            seconds: Pause duration

        Returns:
            This builder instance for chaining
        """
        self.pause_between_keys = seconds
        return self

    def set_release_modifiers_first(self, value: bool = True) -> "KeyUpOptionsBuilder":
        """Set whether to release modifiers first.

        Args:
            value: True to release modifiers first

        Returns:
            This builder instance for chaining
        """
        self.release_modifiers_first = value
        return self

    def build(self) -> KeyUpOptions:
        """Build the immutable KeyUpOptions object.

        Returns:
            A new instance of KeyUpOptions
        """
        return KeyUpOptions(self)

    def _self(self) -> "KeyUpOptionsBuilder":
        """Return self for fluent interface.

        Returns:
            This builder instance
        """
        return self
