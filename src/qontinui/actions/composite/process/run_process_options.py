"""RunProcess action configuration options.

Provides configuration for executing a named process with optional repetition.
"""

from ....actions.action_config import ActionConfig, ActionConfigBuilder
from .process_repetition_options import (
    ProcessRepetitionOptions,
    ProcessRepetitionOptionsBuilder,
)


class RunProcessOptions(ActionConfig):
    """Configuration for RUN_PROCESS action.

    Executes a named process (sequence of actions) with optional repetition
    control including fixed count or until-success modes.
    """

    def __init__(self, builder: "RunProcessOptionsBuilder") -> None:
        """Initialize RunProcessOptions from builder.

        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self.process_id: str = builder.process_id
        self.process_repetition: ProcessRepetitionOptions = builder.process_repetition.build()

    def get_process_id(self) -> str:
        """Get the ID of the process to execute.

        Returns:
            Process ID string
        """
        return self.process_id

    def get_process_repetition(self) -> ProcessRepetitionOptions:
        """Get the process repetition configuration.

        Returns:
            ProcessRepetitionOptions instance
        """
        return self.process_repetition

    @classmethod
    def builder(cls) -> "RunProcessOptionsBuilder":
        """Create a new builder instance.

        Returns:
            New RunProcessOptionsBuilder
        """
        return RunProcessOptionsBuilder()

    def to_builder(self) -> "RunProcessOptionsBuilder":
        """Create a builder from this instance.

        Returns:
            RunProcessOptionsBuilder initialized with current values
        """
        return RunProcessOptionsBuilder(self)


class RunProcessOptionsBuilder(ActionConfigBuilder):
    """Builder for RunProcessOptions using fluent API."""

    def __init__(self, original: RunProcessOptions | None = None) -> None:
        """Initialize builder.

        Args:
            original: Optional RunProcessOptions to copy values from
        """
        super().__init__(original)

        if original:
            self.process_id = original.process_id
            self.process_repetition = original.process_repetition.to_builder()
        else:
            self.process_id = ""
            self.process_repetition = ProcessRepetitionOptionsBuilder()

    def set_process_id(self, process_id: str) -> "RunProcessOptionsBuilder":
        """Set the ID of the process to execute.

        Args:
            process_id: The process ID

        Returns:
            Self for method chaining
        """
        self.process_id = process_id
        return self

    def set_process_repetition(
        self, repetition_builder: ProcessRepetitionOptionsBuilder
    ) -> "RunProcessOptionsBuilder":
        """Set the process repetition configuration.

        Args:
            repetition_builder: ProcessRepetitionOptions builder

        Returns:
            Self for method chaining
        """
        self.process_repetition = repetition_builder
        return self

    def enable_repetition(
        self, max_repeats: int = 10, delay: float = 0.0, until_success: bool = False
    ) -> "RunProcessOptionsBuilder":
        """Enable process repetition with specified parameters.

        Args:
            max_repeats: Maximum additional executions (default: 10)
            delay: Delay between repeats in seconds (default: 0.0)
            until_success: Stop early on success (default: False)

        Returns:
            Self for method chaining
        """
        self.process_repetition = (
            ProcessRepetitionOptionsBuilder()
            .set_enabled(True)
            .set_max_repeats(max_repeats)
            .set_delay(delay)
            .set_until_success(until_success)
        )
        return self

    def build(self) -> RunProcessOptions:
        """Build the RunProcessOptions instance.

        Returns:
            Configured RunProcessOptions
        """
        return RunProcessOptions(self)
