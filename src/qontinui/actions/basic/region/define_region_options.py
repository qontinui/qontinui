"""Define region options - ported from Qontinui framework.

Configuration for region definition actions.
"""

from enum import Enum, auto

from ...action_config import ActionConfig, ActionConfigBuilder


class DefineAs(Enum):
    """How to define a region.

    Port of DefineAs from Qontinui framework enum.
    """

    FOCUSED_WINDOW = auto()  # Define as active window bounds
    MATCH = auto()  # Define as match bounds
    BELOW_MATCH = auto()  # Define below a match
    ABOVE_MATCH = auto()  # Define above a match
    LEFT_OF_MATCH = auto()  # Define to left of match
    RIGHT_OF_MATCH = auto()  # Define to right of match
    INSIDE_ANCHORS = auto()  # Smallest region containing all anchors
    OUTSIDE_ANCHORS = auto()  # Largest region containing all anchors
    INCLUDING_MATCHES = auto()  # Region including all matches


class DefineRegionOptions(ActionConfig):
    """Configuration for region definition actions.

    Port of DefineRegionOptions from Qontinui framework class.

    Configures how to define regions based on various strategies.
    This is an immutable object and must be constructed using its Builder.

    Example usage:
        define_below_match = DefineRegionOptionsBuilder()
            .below_match()
            .with_offset(0, 10)
            .with_expansion(50, 50)
            .build()
    """

    def __init__(self, builder: "DefineRegionOptionsBuilder") -> None:
        """Initialize DefineRegionOptions from builder.

        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self.define_as: DefineAs = builder.define_as
        self.offset_x: int = builder.offset_x
        self.offset_y: int = builder.offset_y
        self.expand_width: int = builder.expand_width
        self.expand_height: int = builder.expand_height

    def get_define_as(self) -> DefineAs:
        """Get how the region should be defined.

        Returns:
            DefineAs enum value
        """
        return self.define_as

    def get_offset_x(self) -> int:
        """Get horizontal offset from base position.

        Returns:
            Horizontal offset in pixels
        """
        return self.offset_x

    def get_offset_y(self) -> int:
        """Get vertical offset from base position.

        Returns:
            Vertical offset in pixels
        """
        return self.offset_y

    def get_expand_width(self) -> int:
        """Get width expansion amount.

        Returns:
            Width expansion in pixels
        """
        return self.expand_width

    def get_expand_height(self) -> int:
        """Get height expansion amount.

        Returns:
            Height expansion in pixels
        """
        return self.expand_height


class DefineRegionOptionsBuilder(ActionConfigBuilder):
    """Builder for constructing DefineRegionOptions with a fluent API.

    Port of DefineRegionOptions from Qontinui framework.Builder.
    """

    def __init__(self, original: DefineRegionOptions | None = None) -> None:
        """Initialize builder.

        Args:
            original: Optional DefineRegionOptions instance to copy values from
        """
        super().__init__(original)

        if original:
            self.define_as = original.define_as
            self.offset_x = original.offset_x
            self.offset_y = original.offset_y
            self.expand_width = original.expand_width
            self.expand_height = original.expand_height
        else:
            self.define_as = DefineAs.MATCH
            self.offset_x = 0
            self.offset_y = 0
            self.expand_width = 0
            self.expand_height = 0

    def as_window(self) -> "DefineRegionOptionsBuilder":
        """Define as focused window.

        Returns:
            This builder instance for chaining
        """
        self.define_as = DefineAs.FOCUSED_WINDOW
        return self

    def as_match(self) -> "DefineRegionOptionsBuilder":
        """Define as match bounds.

        Returns:
            This builder instance for chaining
        """
        self.define_as = DefineAs.MATCH
        return self

    def below_match(self) -> "DefineRegionOptionsBuilder":
        """Define below match.

        Returns:
            This builder instance for chaining
        """
        self.define_as = DefineAs.BELOW_MATCH
        return self

    def above_match(self) -> "DefineRegionOptionsBuilder":
        """Define above match.

        Returns:
            This builder instance for chaining
        """
        self.define_as = DefineAs.ABOVE_MATCH
        return self

    def left_of_match(self) -> "DefineRegionOptionsBuilder":
        """Define to left of match.

        Returns:
            This builder instance for chaining
        """
        self.define_as = DefineAs.LEFT_OF_MATCH
        return self

    def right_of_match(self) -> "DefineRegionOptionsBuilder":
        """Define to right of match.

        Returns:
            This builder instance for chaining
        """
        self.define_as = DefineAs.RIGHT_OF_MATCH
        return self

    def inside_anchors(self) -> "DefineRegionOptionsBuilder":
        """Define as smallest region containing anchors.

        Returns:
            This builder instance for chaining
        """
        self.define_as = DefineAs.INSIDE_ANCHORS
        return self

    def outside_anchors(self) -> "DefineRegionOptionsBuilder":
        """Define as largest region containing anchors.

        Returns:
            This builder instance for chaining
        """
        self.define_as = DefineAs.OUTSIDE_ANCHORS
        return self

    def including_matches(self) -> "DefineRegionOptionsBuilder":
        """Define as region including all matches.

        Returns:
            This builder instance for chaining
        """
        self.define_as = DefineAs.INCLUDING_MATCHES
        return self

    def with_offset(self, x: int, y: int) -> "DefineRegionOptionsBuilder":
        """Set region offset.

        Args:
            x: Horizontal offset
            y: Vertical offset

        Returns:
            This builder instance for chaining
        """
        self.offset_x = x
        self.offset_y = y
        return self

    def with_expansion(self, width: int, height: int) -> "DefineRegionOptionsBuilder":
        """Set region expansion.

        Args:
            width: Width expansion
            height: Height expansion

        Returns:
            This builder instance for chaining
        """
        self.expand_width = width
        self.expand_height = height
        return self

    def build(self) -> DefineRegionOptions:
        """Build the immutable DefineRegionOptions object.

        Returns:
            A new instance of DefineRegionOptions
        """
        return DefineRegionOptions(self)

    def _self(self) -> "DefineRegionOptionsBuilder":
        """Return self for fluent interface.

        Returns:
            This builder instance
        """
        return self
