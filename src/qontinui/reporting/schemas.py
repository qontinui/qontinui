"""Typed event data schemas for qontinui reporting system.

This module provides strongly-typed dataclasses for all event data structures
emitted by the qontinui library. These schemas ensure type safety and provide
clear documentation of event payloads.

Example:
    >>> from qontinui.reporting import register_callback, EventType
    >>> from qontinui.reporting.schemas import MatchAttemptedData
    >>>
    >>> def on_match(event):
    ...     # Type-safe event data access
    ...     data = MatchAttemptedData.from_dict(event.data)
    ...     print(f"Match confidence: {data.best_match_confidence:.2%}")
    ...
    >>> register_callback(EventType.MATCH_ATTEMPTED, on_match)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class EventData(Protocol):
    """Base protocol for all event data types.

    All event data classes should implement this protocol to provide
    consistent serialization/deserialization interfaces.
    """

    version: str

    def to_dict(self) -> dict[str, Any]:
        """Convert event data to dictionary for serialization.

        Returns:
            Dictionary representation of event data
        """
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventData:
        """Create event data from dictionary.

        Args:
            data: Dictionary containing event data

        Returns:
            Typed event data instance
        """
        ...


@dataclass
class MatchAttemptedData:
    """Event data for MATCH_ATTEMPTED events.

    Emitted every time an image/pattern matching operation is attempted,
    regardless of whether the match succeeds or fails. This provides full
    diagnostic visibility into the matching process.

    Attributes:
        image_id: Unique identifier for the image/pattern being matched
        image_name: Human-readable name of the image/pattern (if available)
        template_dimensions: Dimensions of the template being matched
        screenshot_dimensions: Dimensions of the screenshot being searched
        search_region: Region of the screenshot being searched (if specified)
        best_match_location: Location of the best match found
        best_match_confidence: Confidence score of the best match (0.0-1.0)
        similarity_threshold: Minimum confidence required for a match
        threshold_passed: Whether the best match exceeded the threshold
        match_method: Algorithm used for matching (e.g., "CORRELATION_COEFFICIENT_NORMED")
        find_all_mode: Whether finding all matches or just the best match
        version: Schema version for backwards compatibility
    """

    image_id: str
    """Unique identifier for the image/pattern being matched."""

    template_dimensions: dict[str, int]
    """Dimensions of the template being matched (width, height)."""

    screenshot_dimensions: dict[str, int]
    """Dimensions of the screenshot being searched (width, height)."""

    best_match_location: dict[str, Any]
    """Location of the best match found (x, y, region)."""

    best_match_confidence: float
    """Confidence score of the best match (0.0-1.0)."""

    similarity_threshold: float
    """Minimum confidence required for a match."""

    threshold_passed: bool
    """Whether the best match exceeded the threshold."""

    match_method: str
    """Algorithm used for matching (e.g., "CORRELATION_COEFFICIENT_NORMED")."""

    find_all_mode: bool = False
    """Whether finding all matches or just the best match."""

    image_name: str | None = None
    """Human-readable name of the image/pattern (if available)."""

    search_region: dict[str, int] | None = None
    """Region of the screenshot being searched (x, y, width, height) if specified."""

    version: str = "2.0"
    """Schema version for backwards compatibility."""

    def to_dict(self) -> dict[str, Any]:
        """Convert event data to dictionary for serialization.

        Returns:
            Dictionary representation of event data
        """
        return {
            "image_id": self.image_id,
            "image_name": self.image_name,
            "template_dimensions": self.template_dimensions,
            "screenshot_dimensions": self.screenshot_dimensions,
            "search_region": self.search_region,
            "best_match_location": self.best_match_location,
            "best_match_confidence": self.best_match_confidence,
            "similarity_threshold": self.similarity_threshold,
            "threshold_passed": self.threshold_passed,
            "match_method": self.match_method,
            "find_all_mode": self.find_all_mode,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatchAttemptedData:
        """Create event data from dictionary.

        Args:
            data: Dictionary containing event data

        Returns:
            Typed MatchAttemptedData instance
        """
        return cls(
            image_id=data["image_id"],
            image_name=data.get("image_name"),
            template_dimensions=data["template_dimensions"],
            screenshot_dimensions=data["screenshot_dimensions"],
            search_region=data.get("search_region"),
            best_match_location=data["best_match_location"],
            best_match_confidence=data["best_match_confidence"],
            similarity_threshold=data["similarity_threshold"],
            threshold_passed=data["threshold_passed"],
            match_method=data["match_method"],
            find_all_mode=data.get("find_all_mode", False),
            version=data.get("version", "2.0"),
        )


@dataclass
class ActionStartedData:
    """Event data for ACTION_STARTED events.

    Emitted when an action begins execution. Provides context about the
    action being performed and its configuration.

    Attributes:
        action_type: Type of action being executed (e.g., "click", "type", "find")
        action_name: Human-readable name of the action
        target: Target element/pattern for the action (if applicable)
        parameters: Action-specific parameters and configuration
        timestamp: Start timestamp in seconds since epoch
        context_id: Optional context identifier for grouping related actions
        version: Schema version for backwards compatibility
    """

    action_type: str
    """Type of action being executed (e.g., "click", "type", "find")."""

    action_name: str
    """Human-readable name of the action."""

    timestamp: float
    """Start timestamp in seconds since epoch."""

    target: dict[str, Any] | None = None
    """Target element/pattern for the action (if applicable)."""

    parameters: dict[str, Any] = field(default_factory=dict)
    """Action-specific parameters and configuration."""

    context_id: str | None = None
    """Optional context identifier for grouping related actions."""

    version: str = "2.0"
    """Schema version for backwards compatibility."""

    def to_dict(self) -> dict[str, Any]:
        """Convert event data to dictionary for serialization.

        Returns:
            Dictionary representation of event data
        """
        return {
            "action_type": self.action_type,
            "action_name": self.action_name,
            "target": self.target,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "context_id": self.context_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionStartedData:
        """Create event data from dictionary.

        Args:
            data: Dictionary containing event data

        Returns:
            Typed ActionStartedData instance
        """
        return cls(
            action_type=data["action_type"],
            action_name=data["action_name"],
            target=data.get("target"),
            parameters=data.get("parameters", {}),
            timestamp=data["timestamp"],
            context_id=data.get("context_id"),
            version=data.get("version", "2.0"),
        )


@dataclass
class ActionCompletedData:
    """Event data for ACTION_COMPLETED events.

    Emitted when an action completes successfully. Provides the result
    of the action and performance metrics.

    Attributes:
        action_type: Type of action that was executed
        action_name: Human-readable name of the action
        duration: Time taken to execute the action in seconds
        result: Action-specific result data
        success: Whether the action completed successfully
        timestamp: Completion timestamp in seconds since epoch
        context_id: Optional context identifier for grouping related actions
        version: Schema version for backwards compatibility
    """

    action_type: str
    """Type of action that was executed."""

    action_name: str
    """Human-readable name of the action."""

    duration: float
    """Time taken to execute the action in seconds."""

    success: bool
    """Whether the action completed successfully."""

    timestamp: float
    """Completion timestamp in seconds since epoch."""

    result: dict[str, Any] = field(default_factory=dict)
    """Action-specific result data."""

    context_id: str | None = None
    """Optional context identifier for grouping related actions."""

    version: str = "2.0"
    """Schema version for backwards compatibility."""

    def to_dict(self) -> dict[str, Any]:
        """Convert event data to dictionary for serialization.

        Returns:
            Dictionary representation of event data
        """
        return {
            "action_type": self.action_type,
            "action_name": self.action_name,
            "duration": self.duration,
            "result": self.result,
            "success": self.success,
            "timestamp": self.timestamp,
            "context_id": self.context_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionCompletedData:
        """Create event data from dictionary.

        Args:
            data: Dictionary containing event data

        Returns:
            Typed ActionCompletedData instance
        """
        return cls(
            action_type=data["action_type"],
            action_name=data["action_name"],
            duration=data["duration"],
            result=data.get("result", {}),
            success=data["success"],
            timestamp=data["timestamp"],
            context_id=data.get("context_id"),
            version=data.get("version", "2.0"),
        )


@dataclass
class ActionFailedData:
    """Event data for ACTION_FAILED events.

    Emitted when an action fails or encounters an error. Provides detailed
    error information for debugging and monitoring.

    Attributes:
        action_type: Type of action that failed
        action_name: Human-readable name of the action
        error_type: Type/class of the error that occurred
        error_message: Human-readable error message
        duration: Time taken before failure in seconds
        timestamp: Failure timestamp in seconds since epoch
        stack_trace: Optional stack trace for debugging
        error_context: Additional context about the error
        context_id: Optional context identifier for grouping related actions
        version: Schema version for backwards compatibility
    """

    action_type: str
    """Type of action that failed."""

    action_name: str
    """Human-readable name of the action."""

    error_type: str
    """Type/class of the error that occurred."""

    error_message: str
    """Human-readable error message."""

    duration: float
    """Time taken before failure in seconds."""

    timestamp: float
    """Failure timestamp in seconds since epoch."""

    stack_trace: str | None = None
    """Optional stack trace for debugging."""

    error_context: dict[str, Any] = field(default_factory=dict)
    """Additional context about the error."""

    context_id: str | None = None
    """Optional context identifier for grouping related actions."""

    version: str = "2.0"
    """Schema version for backwards compatibility."""

    def to_dict(self) -> dict[str, Any]:
        """Convert event data to dictionary for serialization.

        Returns:
            Dictionary representation of event data
        """
        return {
            "action_type": self.action_type,
            "action_name": self.action_name,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "duration": self.duration,
            "timestamp": self.timestamp,
            "stack_trace": self.stack_trace,
            "error_context": self.error_context,
            "context_id": self.context_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionFailedData:
        """Create event data from dictionary.

        Args:
            data: Dictionary containing event data

        Returns:
            Typed ActionFailedData instance
        """
        return cls(
            action_type=data["action_type"],
            action_name=data["action_name"],
            error_type=data["error_type"],
            error_message=data["error_message"],
            duration=data["duration"],
            timestamp=data["timestamp"],
            stack_trace=data.get("stack_trace"),
            error_context=data.get("error_context", {}),
            context_id=data.get("context_id"),
            version=data.get("version", "2.0"),
        )


# Type aliases for convenience
EventDataType = (
    MatchAttemptedData | ActionStartedData | ActionCompletedData | ActionFailedData
)

__all__ = [
    "EventData",
    "MatchAttemptedData",
    "ActionStartedData",
    "ActionCompletedData",
    "ActionFailedData",
    "EventDataType",
]
