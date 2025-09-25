"""Type options - ported from Qontinui framework.

Configuration for Type actions.
"""

from ...action_config import ActionConfig, ActionConfigBuilder

# Default type delay from Sikuli
DEFAULT_TYPE_DELAY = 0.02


class TypeOptions(ActionConfig):
    """Configuration for Type actions, which send keyboard input.

    Port of TypeOptions from Qontinui framework.

    This class encapsulates all parameters for performing a type action, including
    the delay between keystrokes and any modifier keys (like SHIFT or CTRL) to be held
    during the action.

    It is an immutable object and must be constructed using its inner Builder.
    """

    def __init__(self, builder: "TypeOptionsBuilder"):
        """Initialize TypeOptions from builder.

        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self.type_delay: float = builder.type_delay
        self.modifiers: str = builder.modifiers

    def get_type_delay(self) -> float:
        """Get the delay between keystrokes."""
        return self.type_delay

    def get_modifiers(self) -> str:
        """Get the modifier keys to hold during typing."""
        return self.modifiers


class TypeOptionsBuilder(ActionConfigBuilder):
    """Builder for constructing TypeOptions with a fluent API.

    Port of TypeOptions from Qontinui framework.Builder.
    """

    def __init__(self, original: TypeOptions | None = None):
        """Initialize builder.

        Args:
            original: Optional TypeOptions instance to copy values from
        """
        super().__init__(original)

        if original:
            self.type_delay = original.type_delay
            self.modifiers = original.modifiers
        else:
            self.type_delay = DEFAULT_TYPE_DELAY
            self.modifiers = ""

    def set_type_delay(self, type_delay: float) -> "TypeOptionsBuilder":
        """Set the delay, in seconds, between individual keystrokes.

        Args:
            type_delay: The delay in seconds

        Returns:
            This builder instance for chaining
        """
        self.type_delay = type_delay
        return self

    def set_modifiers(self, modifiers: str) -> "TypeOptionsBuilder":
        """Set the modifier keys to be held down during the type action.

        Sets the modifier keys (e.g., "SHIFT", "CTRL", "ALT") to be held down
        during the type action. Multiple keys can be combined with a "+".

        Args:
            modifiers: A string representing the modifier keys

        Returns:
            This builder instance for chaining
        """
        self.modifiers = modifiers
        return self

    def build(self) -> TypeOptions:
        """Build the immutable TypeOptions object.

        Returns:
            A new instance of TypeOptions
        """
        return TypeOptions(self)

    def _self(self) -> "TypeOptionsBuilder":
        """Return self for fluent interface.

        Returns:
            This builder instance
        """
        return self
