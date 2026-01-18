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


# =============================================================================
# Healing Event Data Schemas
# =============================================================================


@dataclass
class HealingCacheEventData:
    """Event data for healing cache events (hit/miss/invalidated).

    Emitted when cache operations occur during element lookup.

    Attributes:
        pattern_id: Unique identifier of the cached pattern
        pattern_name: Human-readable pattern name (if available)
        cache_hit: Whether this was a cache hit (True) or miss (False)
        cache_size: Current number of items in cache
        hit_rate: Overall cache hit rate (0.0-1.0)
        total_hits: Total cache hits since start
        total_misses: Total cache misses since start
        timestamp: Event timestamp
        version: Schema version for backwards compatibility
    """

    pattern_id: str
    """Unique identifier of the cached pattern."""

    cache_hit: bool
    """Whether this was a cache hit (True) or miss (False)."""

    timestamp: float
    """Event timestamp in seconds since epoch."""

    pattern_name: str | None = None
    """Human-readable pattern name (if available)."""

    cache_size: int = 0
    """Current number of items in cache."""

    hit_rate: float = 0.0
    """Overall cache hit rate (0.0-1.0)."""

    total_hits: int = 0
    """Total cache hits since start."""

    total_misses: int = 0
    """Total cache misses since start."""

    version: str = "1.0"
    """Schema version for backwards compatibility."""

    def to_dict(self) -> dict[str, Any]:
        """Convert event data to dictionary for serialization.

        Returns:
            Dictionary representation of event data
        """
        return {
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern_name,
            "cache_hit": self.cache_hit,
            "cache_size": self.cache_size,
            "hit_rate": self.hit_rate,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "timestamp": self.timestamp,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HealingCacheEventData:
        """Create event data from dictionary.

        Args:
            data: Dictionary containing event data

        Returns:
            Typed HealingCacheEventData instance
        """
        return cls(
            pattern_id=data["pattern_id"],
            pattern_name=data.get("pattern_name"),
            cache_hit=data["cache_hit"],
            cache_size=data.get("cache_size", 0),
            hit_rate=data.get("hit_rate", 0.0),
            total_hits=data.get("total_hits", 0),
            total_misses=data.get("total_misses", 0),
            timestamp=data["timestamp"],
            version=data.get("version", "1.0"),
        )


@dataclass
class HealingAttemptData:
    """Event data for healing attempt events (started/succeeded/failed).

    Emitted when a healing operation is attempted.

    Attributes:
        pattern_id: ID of pattern being healed
        pattern_name: Human-readable pattern name
        strategy: Strategy being used (visual_search, llm_vision, etc.)
        success: Whether the healing attempt succeeded
        confidence: Match confidence if successful (0.0-1.0)
        duration_ms: Time taken for the healing attempt
        error_message: Error message if failed
        location_x: X coordinate if element found
        location_y: Y coordinate if element found
        timestamp: Event timestamp
        context_id: Optional context identifier
        version: Schema version
    """

    pattern_id: str
    """ID of pattern being healed."""

    strategy: str
    """Strategy being used (visual_search, llm_vision, etc.)."""

    success: bool
    """Whether the healing attempt succeeded."""

    timestamp: float
    """Event timestamp in seconds since epoch."""

    pattern_name: str | None = None
    """Human-readable pattern name."""

    confidence: float = 0.0
    """Match confidence if successful (0.0-1.0)."""

    duration_ms: float = 0.0
    """Time taken for the healing attempt in milliseconds."""

    error_message: str | None = None
    """Error message if failed."""

    location_x: int | None = None
    """X coordinate if element found."""

    location_y: int | None = None
    """Y coordinate if element found."""

    context_id: str | None = None
    """Optional context identifier for grouping."""

    version: str = "1.0"
    """Schema version for backwards compatibility."""

    def to_dict(self) -> dict[str, Any]:
        """Convert event data to dictionary for serialization.

        Returns:
            Dictionary representation of event data
        """
        return {
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern_name,
            "strategy": self.strategy,
            "success": self.success,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "location_x": self.location_x,
            "location_y": self.location_y,
            "timestamp": self.timestamp,
            "context_id": self.context_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HealingAttemptData:
        """Create event data from dictionary.

        Args:
            data: Dictionary containing event data

        Returns:
            Typed HealingAttemptData instance
        """
        return cls(
            pattern_id=data["pattern_id"],
            pattern_name=data.get("pattern_name"),
            strategy=data["strategy"],
            success=data["success"],
            confidence=data.get("confidence", 0.0),
            duration_ms=data.get("duration_ms", 0.0),
            error_message=data.get("error_message"),
            location_x=data.get("location_x"),
            location_y=data.get("location_y"),
            timestamp=data["timestamp"],
            context_id=data.get("context_id"),
            version=data.get("version", "1.0"),
        )


@dataclass
class HealingMetricsData:
    """Event data for aggregate healing metrics updates.

    Emitted periodically or on significant metric changes.

    Attributes:
        total_attempts: Total healing attempts
        successful_heals: Number of successful heals
        failed_heals: Number of failed heals
        healing_rate: Success rate (0.0-1.0)
        avg_healing_time_ms: Average healing time
        patterns_updated: Number of patterns auto-updated
        llm_calls: Number of LLM healing calls
        cache_hit_rate: Cache hit rate (0.0-1.0)
        timestamp: Event timestamp
        version: Schema version
    """

    total_attempts: int
    """Total healing attempts."""

    successful_heals: int
    """Number of successful heals."""

    failed_heals: int
    """Number of failed heals."""

    healing_rate: float
    """Success rate (0.0-1.0)."""

    timestamp: float
    """Event timestamp in seconds since epoch."""

    avg_healing_time_ms: float = 0.0
    """Average healing time in milliseconds."""

    patterns_updated: int = 0
    """Number of patterns auto-updated."""

    llm_calls: int = 0
    """Number of LLM healing calls."""

    cache_hit_rate: float = 0.0
    """Cache hit rate (0.0-1.0)."""

    version: str = "1.0"
    """Schema version for backwards compatibility."""

    def to_dict(self) -> dict[str, Any]:
        """Convert event data to dictionary for serialization.

        Returns:
            Dictionary representation of event data
        """
        return {
            "total_attempts": self.total_attempts,
            "successful_heals": self.successful_heals,
            "failed_heals": self.failed_heals,
            "healing_rate": self.healing_rate,
            "avg_healing_time_ms": self.avg_healing_time_ms,
            "patterns_updated": self.patterns_updated,
            "llm_calls": self.llm_calls,
            "cache_hit_rate": self.cache_hit_rate,
            "timestamp": self.timestamp,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HealingMetricsData:
        """Create event data from dictionary.

        Args:
            data: Dictionary containing event data

        Returns:
            Typed HealingMetricsData instance
        """
        return cls(
            total_attempts=data["total_attempts"],
            successful_heals=data["successful_heals"],
            failed_heals=data["failed_heals"],
            healing_rate=data["healing_rate"],
            avg_healing_time_ms=data.get("avg_healing_time_ms", 0.0),
            patterns_updated=data.get("patterns_updated", 0),
            llm_calls=data.get("llm_calls", 0),
            cache_hit_rate=data.get("cache_hit_rate", 0.0),
            timestamp=data["timestamp"],
            version=data.get("version", "1.0"),
        )


@dataclass
class VisualValidationData:
    """Event data for visual validation events.

    Emitted when visual validation is performed to verify element state.

    Attributes:
        validation_type: Type of validation performed
        expected_state: Expected state identifier
        actual_state: Actually detected state
        confidence: Validation confidence (0.0-1.0)
        threshold: Confidence threshold used
        passed: Whether validation passed
        pattern_id: ID of pattern validated
        screenshot_reference: Reference to screenshot used
        timestamp: Event timestamp
        version: Schema version
    """

    validation_type: str
    """Type of validation performed."""

    passed: bool
    """Whether validation passed."""

    confidence: float
    """Validation confidence (0.0-1.0)."""

    threshold: float
    """Confidence threshold used."""

    timestamp: float
    """Event timestamp in seconds since epoch."""

    expected_state: str | None = None
    """Expected state identifier."""

    actual_state: str | None = None
    """Actually detected state."""

    pattern_id: str | None = None
    """ID of pattern validated."""

    screenshot_reference: str | None = None
    """Reference to screenshot used."""

    version: str = "1.0"
    """Schema version for backwards compatibility."""

    def to_dict(self) -> dict[str, Any]:
        """Convert event data to dictionary for serialization.

        Returns:
            Dictionary representation of event data
        """
        return {
            "validation_type": self.validation_type,
            "expected_state": self.expected_state,
            "actual_state": self.actual_state,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "passed": self.passed,
            "pattern_id": self.pattern_id,
            "screenshot_reference": self.screenshot_reference,
            "timestamp": self.timestamp,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisualValidationData:
        """Create event data from dictionary.

        Args:
            data: Dictionary containing event data

        Returns:
            Typed VisualValidationData instance
        """
        return cls(
            validation_type=data["validation_type"],
            expected_state=data.get("expected_state"),
            actual_state=data.get("actual_state"),
            confidence=data["confidence"],
            threshold=data["threshold"],
            passed=data["passed"],
            pattern_id=data.get("pattern_id"),
            screenshot_reference=data.get("screenshot_reference"),
            timestamp=data["timestamp"],
            version=data.get("version", "1.0"),
        )


@dataclass
class ReliabilityChangeData:
    """Event data for reliability score changes.

    Emitted when a pattern's reliability score changes significantly.

    Attributes:
        pattern_id: ID of the pattern
        pattern_name: Human-readable pattern name
        new_score: New reliability score (0.0-1.0)
        old_score: Previous reliability score (0.0-1.0)
        change_amount: Absolute change in score
        total_uses: Total times this pattern was used
        successful_uses: Successful matches for this pattern
        timestamp: Event timestamp
        version: Schema version
    """

    pattern_id: str
    """ID of the pattern."""

    new_score: float
    """New reliability score (0.0-1.0)."""

    old_score: float
    """Previous reliability score (0.0-1.0)."""

    timestamp: float
    """Event timestamp in seconds since epoch."""

    pattern_name: str | None = None
    """Human-readable pattern name."""

    change_amount: float = 0.0
    """Absolute change in score."""

    total_uses: int = 0
    """Total times this pattern was used."""

    successful_uses: int = 0
    """Successful matches for this pattern."""

    version: str = "1.0"
    """Schema version for backwards compatibility."""

    def to_dict(self) -> dict[str, Any]:
        """Convert event data to dictionary for serialization.

        Returns:
            Dictionary representation of event data
        """
        return {
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern_name,
            "new_score": self.new_score,
            "old_score": self.old_score,
            "change_amount": self.change_amount,
            "total_uses": self.total_uses,
            "successful_uses": self.successful_uses,
            "timestamp": self.timestamp,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReliabilityChangeData:
        """Create event data from dictionary.

        Args:
            data: Dictionary containing event data

        Returns:
            Typed ReliabilityChangeData instance
        """
        return cls(
            pattern_id=data["pattern_id"],
            pattern_name=data.get("pattern_name"),
            new_score=data["new_score"],
            old_score=data["old_score"],
            change_amount=data.get("change_amount", abs(data["new_score"] - data["old_score"])),
            total_uses=data.get("total_uses", 0),
            successful_uses=data.get("successful_uses", 0),
            timestamp=data["timestamp"],
            version=data.get("version", "1.0"),
        )


# Type aliases for convenience
EventDataType = (
    MatchAttemptedData
    | ActionStartedData
    | ActionCompletedData
    | ActionFailedData
    | HealingCacheEventData
    | HealingAttemptData
    | HealingMetricsData
    | VisualValidationData
    | ReliabilityChangeData
)

__all__ = [
    "EventData",
    "MatchAttemptedData",
    "ActionStartedData",
    "ActionCompletedData",
    "ActionFailedData",
    # Healing event schemas
    "HealingCacheEventData",
    "HealingAttemptData",
    "HealingMetricsData",
    "VisualValidationData",
    "ReliabilityChangeData",
    "EventDataType",
]
