"""Highlight options - ported from Qontinui framework.

Configuration for highlight actions.
"""

from ...action_config import ActionConfig, ActionConfigBuilder


class HighlightOptions(ActionConfig):
    """Configuration for highlight actions.

    Port of HighlightOptions from Qontinui framework.

    This class encapsulates parameters for highlighting regions on screen,
    typically used for debugging or user feedback.
    """

    def __init__(self, builder: "HighlightOptionsBuilder"):
        """Initialize HighlightOptions from builder.

        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self.highlight_duration: float = builder.highlight_duration
        self.color: tuple[int, int, int] = builder.color
        self.thickness: int = builder.thickness
        self.flash: bool = builder.flash
        self.flash_times: int = builder.flash_times

    def get_highlight_duration(self) -> float:
        """Get the duration to show the highlight in seconds."""
        return self.highlight_duration

    def get_color(self) -> tuple[int, int, int]:
        """Get the highlight color as RGB tuple."""
        return self.color

    def get_thickness(self) -> int:
        """Get the highlight border thickness in pixels."""
        return self.thickness

    def is_flash(self) -> bool:
        """Check if the highlight should flash."""
        return self.flash

    def get_flash_times(self) -> int:
        """Get the number of times to flash."""
        return self.flash_times


class HighlightOptionsBuilder(ActionConfigBuilder):
    """Builder for constructing HighlightOptions with a fluent API.

    Port of HighlightOptions from Qontinui framework.Builder.
    """

    def __init__(self, original: HighlightOptions | None = None):
        """Initialize builder.

        Args:
            original: Optional HighlightOptions instance to copy values from
        """
        super().__init__(original)

        if original:
            self.highlight_duration = original.highlight_duration
            self.color = original.color
            self.thickness = original.thickness
            self.flash = original.flash
            self.flash_times = original.flash_times
        else:
            self.highlight_duration = 2.0
            self.color = (255, 0, 0)  # Red
            self.thickness = 3
            self.flash = False
            self.flash_times = 3

    def set_highlight_duration(self, duration: float) -> "HighlightOptionsBuilder":
        """Set the highlight duration.

        Args:
            duration: Duration in seconds to show the highlight

        Returns:
            This builder instance for chaining
        """
        self.highlight_duration = duration
        return self

    def set_color(self, red: int, green: int, blue: int) -> "HighlightOptionsBuilder":
        """Set the highlight color.

        Args:
            red: Red component (0-255)
            green: Green component (0-255)
            blue: Blue component (0-255)

        Returns:
            This builder instance for chaining
        """
        self.color = (red, green, blue)
        return self

    def set_thickness(self, thickness: int) -> "HighlightOptionsBuilder":
        """Set the highlight border thickness.

        Args:
            thickness: Thickness in pixels

        Returns:
            This builder instance for chaining
        """
        self.thickness = thickness
        return self

    def set_flash(self, flash: bool) -> "HighlightOptionsBuilder":
        """Enable or disable flashing.

        Args:
            flash: Whether the highlight should flash

        Returns:
            This builder instance for chaining
        """
        self.flash = flash
        return self

    def set_flash_times(self, times: int) -> "HighlightOptionsBuilder":
        """Set the number of times to flash.

        Args:
            times: Number of flash cycles

        Returns:
            This builder instance for chaining
        """
        self.flash_times = times
        return self

    def build(self) -> HighlightOptions:
        """Build the immutable HighlightOptions object.

        Returns:
            A new instance of HighlightOptions
        """
        return HighlightOptions(self)

    def _self(self) -> "HighlightOptionsBuilder":
        """Return self for fluent interface.

        Returns:
            This builder instance
        """
        return self
