"""Process repetition configuration options.

Controls how many times a process is repeated and under what conditions.
"""

from dataclasses import dataclass


@dataclass
class ProcessRepetitionOptions:
    """Configuration for process repetition behavior.

    Supports two modes:
    1. Fixed count: Run exactly maxRepeats additional times
    2. Until success: Stop early on success, otherwise run up to maxRepeats
    """

    enabled: bool = False
    """Whether process repetition is active."""

    max_repeats: int = 10
    """Maximum number of additional repeats (1 = run once more).

    When untilSuccess is False: Runs exactly this many additional times.
    When untilSuccess is True: Upper limit, stops early on success.
    """

    delay: float = 0.0
    """Delay between repeats in seconds (converted from milliseconds in JSON)."""

    until_success: bool = False
    """If True, stop early on success; if False, run all maxRepeats."""

    @classmethod
    def builder(cls) -> "ProcessRepetitionOptionsBuilder":
        """Create a new builder instance.

        Returns:
            New ProcessRepetitionOptionsBuilder
        """
        return ProcessRepetitionOptionsBuilder()

    def to_builder(self) -> "ProcessRepetitionOptionsBuilder":
        """Create a builder from this instance.

        Returns:
            ProcessRepetitionOptionsBuilder initialized with current values
        """
        return ProcessRepetitionOptionsBuilder(self)

    def get_enabled(self) -> bool:
        """Check if repetition is enabled."""
        return self.enabled

    def get_max_repeats(self) -> int:
        """Get maximum number of repeats."""
        return self.max_repeats

    def get_delay(self) -> float:
        """Get delay between repeats in seconds."""
        return self.delay

    def get_until_success(self) -> bool:
        """Check if should stop early on success."""
        return self.until_success


class ProcessRepetitionOptionsBuilder:
    """Builder for ProcessRepetitionOptions using fluent API."""

    def __init__(self, original: ProcessRepetitionOptions | None = None):
        """Initialize builder.

        Args:
            original: Optional ProcessRepetitionOptions to copy values from
        """
        if original:
            self.enabled = original.enabled
            self.max_repeats = original.max_repeats
            self.delay = original.delay
            self.until_success = original.until_success
        else:
            self.enabled = False
            self.max_repeats = 10
            self.delay = 0.0
            self.until_success = False

    def set_enabled(self, enabled: bool) -> "ProcessRepetitionOptionsBuilder":
        """Set whether repetition is enabled.

        Args:
            enabled: True to enable repetition

        Returns:
            Self for method chaining
        """
        self.enabled = enabled
        return self

    def set_max_repeats(self, max_repeats: int) -> "ProcessRepetitionOptionsBuilder":
        """Set maximum number of repeats.

        Args:
            max_repeats: Maximum additional executions (1 = run once more)

        Returns:
            Self for method chaining
        """
        self.max_repeats = max_repeats
        return self

    def set_delay(self, delay: float) -> "ProcessRepetitionOptionsBuilder":
        """Set delay between repeats.

        Args:
            delay: Delay in seconds

        Returns:
            Self for method chaining
        """
        self.delay = delay
        return self

    def set_until_success(self, until_success: bool) -> "ProcessRepetitionOptionsBuilder":
        """Set whether to stop early on success.

        Args:
            until_success: If True, stop on first success; if False, run all

        Returns:
            Self for method chaining
        """
        self.until_success = until_success
        return self

    def build(self) -> ProcessRepetitionOptions:
        """Build the ProcessRepetitionOptions instance.

        Returns:
            Configured ProcessRepetitionOptions
        """
        return ProcessRepetitionOptions(
            enabled=self.enabled,
            max_repeats=self.max_repeats,
            delay=self.delay,
            until_success=self.until_success,
        )
