"""Exceptions - ported from Qontinui framework.

Custom exception hierarchy for the Qontinui framework.
"""

from typing import Any

from ..action_type import ActionType


class QontinuiException(Exception):
    """Base exception for all Qontinui exceptions.

    Port of base from Qontinui framework exception hierarchy.

    All framework-specific exceptions inherit from this class,
    allowing for unified exception handling throughout the system.
    """

    def __init__(self, message: str, cause: Exception | None = None):
        """Initialize exception.

        Args:
            message: Exception message
            cause: Underlying cause exception
        """
        super().__init__(message)
        self.cause = cause


class QontinuiRuntimeException(QontinuiException):
    """Runtime exception for Brobot/Qontinui operations.

    Port of QontinuiRuntimeException from Qontinui framework.

    Base class for runtime exceptions that occur during
    automation execution.
    """

    pass


class ActionFailedException(QontinuiRuntimeException):
    """Thrown when an action fails during execution.

    Port of ActionFailedException from Qontinui framework.

    This exception indicates that a specific action (Click, Find, Type, etc.)
    could not complete successfully. It provides context about which action failed
    and why, enabling the framework to make intelligent decisions about recovery
    strategies.
    """

    def __init__(self, action_type: ActionType, message: str, cause: Exception | None = None):
        """Construct a new action failed exception.

        Args:
            action_type: The type of action that failed
            message: A description of why the action failed
            cause: The underlying cause of the failure
        """
        super().__init__(f"Action {action_type.name} failed: {message}", cause)
        self.action_type = action_type
        self.action_details = message

    def get_action_type(self) -> ActionType:
        """Get the type of action that failed.

        Returns:
            The action type
        """
        return self.action_type

    def get_action_details(self) -> str:
        """Get detailed information about why the action failed.

        Returns:
            The failure details
        """
        return self.action_details


class StateNotFoundException(QontinuiRuntimeException):
    """Thrown when a requested state cannot be found.

    Port of StateNotFoundException from Qontinui framework.

    This exception indicates that a state referenced in a transition
    or navigation request does not exist in the state registry.
    """

    def __init__(self, state_name: str, cause: Exception | None = None):
        """Construct a new state not found exception.

        Args:
            state_name: Name of the state that was not found
            cause: The underlying cause
        """
        super().__init__(f"State not found: {state_name}", cause)
        self.state_name = state_name

    def get_state_name(self) -> str:
        """Get the name of the state that was not found.

        Returns:
            The state name
        """
        return self.state_name


class ConfigurationException(QontinuiRuntimeException):
    """Thrown when there is a configuration error.

    Port of ConfigurationException from Qontinui framework.

    This exception indicates a problem with framework configuration,
    such as invalid settings, missing required configuration, or
    conflicting configuration values.
    """

    def __init__(self, message: str, config_key: str | None = None, cause: Exception | None = None):
        """Construct a new configuration exception.

        Args:
            message: Description of the configuration error
            config_key: The configuration key that caused the error
            cause: The underlying cause
        """
        if config_key:
            full_message = f"Configuration error for '{config_key}': {message}"
        else:
            full_message = f"Configuration error: {message}"

        super().__init__(full_message, cause)
        self.config_key = config_key

    def get_config_key(self) -> str | None:
        """Get the configuration key that caused the error.

        Returns:
            The configuration key or None
        """
        return self.config_key


class FindFailedException(ActionFailedException):
    """Thrown when a find operation fails.

    Specialized exception for find operations that provides
    additional context about what was being searched for.
    """

    def __init__(
        self,
        target: str,
        search_region: str | None = None,
        message: str | None = None,
        cause: Exception | None = None,
    ):
        """Construct a new find failed exception.

        Args:
            target: What was being searched for
            search_region: Where the search was performed
            message: Additional error message
            cause: The underlying cause
        """
        if message:
            full_message = message
        else:
            full_message = f"Could not find '{target}'"
            if search_region:
                full_message += f" in region {search_region}"

        super().__init__(ActionType.FIND, full_message, cause)
        self.target = target
        self.search_region = search_region

    def get_target(self) -> str:
        """Get what was being searched for.

        Returns:
            The search target
        """
        return self.target

    def get_search_region(self) -> str | None:
        """Get where the search was performed.

        Returns:
            The search region or None
        """
        return self.search_region


class ImageNotFoundException(FindFailedException):
    """Thrown when an image cannot be found on screen.

    Specialized exception for image-based find operations.
    """

    def __init__(
        self,
        image_name: str,
        similarity: float = 0.7,
        search_region: str | None = None,
        cause: Exception | None = None,
    ):
        """Construct a new image not found exception.

        Args:
            image_name: Name of the image that was not found
            similarity: Similarity threshold used
            search_region: Where the search was performed
            cause: The underlying cause
        """
        message = f"Image '{image_name}' not found (similarity threshold: {similarity})"
        super().__init__(image_name, search_region, message, cause)
        self.image_name = image_name
        self.similarity = similarity


class TextNotFoundException(FindFailedException):
    """Thrown when text cannot be found on screen.

    Specialized exception for text-based find operations.
    """

    def __init__(self, text: str, search_region: str | None = None, cause: Exception | None = None):
        """Construct a new text not found exception.

        Args:
            text: Text that was not found
            search_region: Where the search was performed
            cause: The underlying cause
        """
        message = f"Text '{text}' not found"
        super().__init__(text, search_region, message, cause)
        self.text = text


class TimeoutException(QontinuiRuntimeException):
    """Thrown when an operation times out.

    This exception indicates that an operation took longer
    than the allowed timeout period.
    """

    def __init__(self, operation: str, timeout_seconds: float, cause: Exception | None = None):
        """Construct a new timeout exception.

        Args:
            operation: Description of the operation that timed out
            timeout_seconds: The timeout period that was exceeded
            cause: The underlying cause
        """
        message = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        super().__init__(message, cause)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class ValidationException(QontinuiRuntimeException):
    """Thrown when validation fails.

    This exception indicates that input validation or
    state validation has failed.
    """

    def __init__(self, field: str, value: Any, reason: str, cause: Exception | None = None):
        """Construct a new validation exception.

        Args:
            field: Field that failed validation
            value: The invalid value
            reason: Why validation failed
            cause: The underlying cause
        """
        message = f"Validation failed for '{field}': {reason} (value: {value})"
        super().__init__(message, cause)
        self.field = field
        self.value = value
        self.reason = reason
