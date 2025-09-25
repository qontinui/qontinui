"""Action configuration - ported from Qontinui framework.

Base configuration for all actions.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class Illustrate(Enum):
    """Override the global illustration setting for a specific action.

    Port of ActionConfig from Qontinui framework.Illustrate enum.
    """

    YES = auto()
    """Always generate an illustration for this action."""

    NO = auto()
    """Never generate an illustration for this action."""

    USE_GLOBAL = auto()
    """Use the global framework setting to decide. This is the default."""


@dataclass
class LoggingOptions:
    """Configuration for automatic logging of action results.

    Port of ActionConfig from Qontinui framework.LoggingOptions class.

    Enables streamlined success/failure logging without manual checks.
    """

    before_action_message: str | None = None
    after_action_message: str | None = None
    success_message: str | None = None
    failure_message: str | None = None
    log_before_action: bool = False
    log_after_action: bool = False
    log_on_success: bool = True
    log_on_failure: bool = True
    before_action_level: str = "ACTION"
    after_action_level: str = "ACTION"
    success_level: str = "ACTION"
    failure_level: str = "ERROR"

    @classmethod
    def defaults(cls) -> "LoggingOptions":
        """Create default logging options with standard messages.

        Returns:
            Default LoggingOptions instance
        """
        return cls()


class ActionConfig:
    """Abstract base class for all action configurations.

    Port of ActionConfig from Qontinui framework class.

    This class defines the common parameters that are applicable to any action,
    such as timing, success criteria, and illustration settings. It uses a generic,
    inheritable Builder pattern to ensure that all specialized configuration classes
    can provide a consistent and fluent API.

    Specialized configuration classes (e.g., ClickOptions, FindOptions)
    must extend this class.
    """

    def __init__(self, builder: Optional["ActionConfigBuilder"] = None):
        """Initialize ActionConfig from builder.

        Args:
            builder: The builder instance containing configuration values
        """
        if builder is None:
            # Default values
            self.pause_before_begin = 0.0
            self.pause_after_end = 0.0
            self.success_criteria = None
            self.illustrate = Illustrate.USE_GLOBAL
            self.subsequent_actions = []
            self.log_type = "ACTION"
            self.logging_options = LoggingOptions.defaults()
        else:
            self.pause_before_begin = builder.pause_before_begin
            self.pause_after_end = builder.pause_after_end
            self.success_criteria = builder.success_criteria
            self.illustrate = builder.illustrate
            self.subsequent_actions = builder.subsequent_actions.copy()
            self.log_type = builder.log_type
            self.logging_options = builder.logging_options

    def get_pause_before_begin(self) -> float:
        """Get pause duration before action begins.

        Returns:
            Pause duration in seconds
        """
        return self.pause_before_begin

    def get_pause_after_end(self) -> float:
        """Get pause duration after action ends.

        Returns:
            Pause duration in seconds
        """
        return self.pause_after_end

    def get_success_criteria(self) -> Callable[["ActionResult"], bool] | None:
        """Get custom success criteria predicate.

        Returns:
            Success criteria function or None
        """
        return self.success_criteria

    def get_illustrate(self) -> Illustrate:
        """Get illustration setting.

        Returns:
            Illustrate enum value
        """
        return self.illustrate

    def get_subsequent_actions(self) -> list["ActionConfig"]:
        """Get list of subsequent actions to execute.

        Returns:
            List of ActionConfig instances
        """
        return self.subsequent_actions

    def get_log_type(self) -> str:
        """Get log event type.

        Returns:
            Log type string
        """
        return self.log_type

    def get_logging_options(self) -> LoggingOptions:
        """Get logging options.

        Returns:
            LoggingOptions instance
        """
        return self.logging_options


class ActionConfigBuilder(ABC):
    """Abstract generic builder for constructing ActionConfig and its subclasses.

    Port of ActionConfig from Qontinui framework.Builder class.

    This pattern allows for fluent, inheritable builder methods.

    Example of chaining actions:
        find_and_click = PatternFindOptionsBuilder()\\
            .set_strategy(Strategy.BEST)\\
            .set_pause_after_end(0.5)\\
            .then(ClickOptionsBuilder()\\
                .set_number_of_clicks(1)\\
                .build())\\
            .build()
    """

    def __init__(self, original: ActionConfig | None = None):
        """Initialize builder, optionally from existing config.

        Args:
            original: Optional ActionConfig to copy values from
        """
        if original:
            self.pause_before_begin = original.pause_before_begin
            self.pause_after_end = original.pause_after_end
            self.success_criteria = original.success_criteria
            self.illustrate = original.illustrate
            self.subsequent_actions = original.subsequent_actions.copy()
            self.log_type = original.log_type
            self.logging_options = original.logging_options
        else:
            self.pause_before_begin = 0.0
            self.pause_after_end = 0.0
            self.success_criteria = None
            self.illustrate = Illustrate.USE_GLOBAL
            self.subsequent_actions = []
            self.log_type = "ACTION"
            self.logging_options = LoggingOptions.defaults()

    def set_pause_before_begin(self, seconds: float) -> "ActionConfigBuilder":
        """Set pause duration before action begins.

        Args:
            seconds: Pause duration in seconds

        Returns:
            This builder for method chaining
        """
        self.pause_before_begin = seconds
        return self

    def set_pause_after_end(self, seconds: float) -> "ActionConfigBuilder":
        """Set pause duration after action completes.

        Args:
            seconds: Pause duration in seconds

        Returns:
            This builder for method chaining
        """
        self.pause_after_end = seconds
        return self

    def set_success_criteria(
        self, criteria: Callable[["ActionResult"], bool]
    ) -> "ActionConfigBuilder":
        """Set custom success evaluation predicate.

        This overrides all default success evaluation logic for the action.

        Args:
            criteria: Predicate that takes ActionResult and returns True for success

        Returns:
            This builder for method chaining
        """
        self.success_criteria = criteria
        return self

    def set_illustrate(self, illustrate: Illustrate) -> "ActionConfigBuilder":
        """Override global illustration setting.

        Args:
            illustrate: Illustration override setting

        Returns:
            This builder for method chaining
        """
        self.illustrate = illustrate
        return self

    def set_log_type(self, log_type: str) -> "ActionConfigBuilder":
        """Set log event type for categorizing this action.

        Args:
            log_type: Type of log event this action represents

        Returns:
            This builder for method chaining
        """
        self.log_type = log_type
        return self

    def then(self, next_action_config: ActionConfig) -> "ActionConfigBuilder":
        """Chain another action to execute after this one.

        The subsequent action will operate on the results of this action.

        Args:
            next_action_config: Configuration of the action to execute next

        Returns:
            This builder for method chaining
        """
        self.subsequent_actions.append(next_action_config)
        return self

    def with_before_action_log(self, message: str) -> "ActionConfigBuilder":
        """Set message to log before action begins.

        Args:
            message: Message to log before execution

        Returns:
            This builder for method chaining
        """
        self.logging_options.before_action_message = message
        self.logging_options.log_before_action = True
        return self

    def with_after_action_log(self, message: str) -> "ActionConfigBuilder":
        """Set message to log after action completes.

        Args:
            message: Message to log after execution

        Returns:
            This builder for method chaining
        """
        self.logging_options.after_action_message = message
        self.logging_options.log_after_action = True
        return self

    def with_success_log(self, message: str) -> "ActionConfigBuilder":
        """Set message to log on successful completion.

        Args:
            message: Message to log on success

        Returns:
            This builder for method chaining
        """
        self.logging_options.success_message = message
        self.logging_options.log_on_success = True
        return self

    def with_failure_log(self, message: str) -> "ActionConfigBuilder":
        """Set message to log on failure.

        Args:
            message: Message to log on failure

        Returns:
            This builder for method chaining
        """
        self.logging_options.failure_message = message
        self.logging_options.log_on_failure = True
        return self

    def set_logging_options(self, logging_options: LoggingOptions) -> "ActionConfigBuilder":
        """Set complete logging options.

        Args:
            logging_options: LoggingOptions instance

        Returns:
            This builder for method chaining
        """
        self.logging_options = logging_options
        return self

    @abstractmethod
    def build(self) -> ActionConfig:
        """Build the ActionConfig instance.

        Must be implemented by concrete builder subclasses.

        Returns:
            Configured ActionConfig instance
        """
        pass


# Forward reference
class ActionResult:
    """Placeholder for ActionResult class."""

    pass
