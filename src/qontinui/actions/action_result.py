"""Action result - ported from Qontinui framework.

Immutable results container for action executions.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..find.match import Match
    from .action_config import ActionConfig

from ..model.element.region import Region


@dataclass(frozen=True)
class ActionResult:
    """Immutable result of an action execution.

    This is a clean, immutable data class that represents the final result
    of an action. Construction is handled by ActionResultBuilder.

    Thread Safety:
        Immutable after creation - safe to share across threads without locks.
        All collections are immutable tuples/frozensets.

    Design Philosophy:
        - No optional fields that could be None - use empty collections instead
        - No mutation methods - use builder pattern to construct
        - Explicit about what data exists vs doesn't exist
        - Clear ownership - result is created once and never modified
    """

    success: bool
    """Whether the action achieved its intended goal."""

    matches: tuple[Match, ...]
    """All matches found during action execution. Empty tuple if none."""

    times_acted_on: int
    """Count of how many times this object was acted upon."""

    text: str
    """Accumulated text content from all matches. Empty string if none."""

    defined_regions: tuple[Region, ...]
    """Regions created or captured by DEFINE actions. Empty tuple if none."""

    movements: tuple[Movement, ...]
    """List of movements performed during action execution. Empty tuple if none."""

    execution_history: tuple[ExecutionRecord, ...]
    """Ordered history of action execution steps. Empty tuple if none."""

    active_states: frozenset[str]
    """Names of states identified as active during action execution. Empty set if none."""

    action_description: str = ""
    """Human-readable description of the action performed."""

    output_text: str = ""
    """Formatted text output for reporting and logging."""

    selected_text: str = ""
    """Specific text selected or highlighted during the action."""

    duration: timedelta | None = None
    """Total time taken for action execution. None if not measured."""

    start_time: datetime | None = None
    """Timestamp when action execution began. None if not recorded."""

    end_time: datetime | None = None
    """Timestamp when action execution completed. None if not recorded."""

    action_config: ActionConfig | None = None
    """Configuration used for this action execution. None if not provided."""

    @property
    def is_success(self) -> bool:
        """Check if action was successful.

        Returns:
            True if action succeeded
        """
        return self.success

    @property
    def match_count(self) -> int:
        """Get the number of matches.

        Returns:
            Count of matches
        """
        return len(self.matches)

    def __str__(self) -> str:
        """String representation for debugging.

        Returns:
            Human-readable string
        """
        result = f"ActionResult: success={self.success}, matches={len(self.matches)}"
        if self.action_description:
            result += f", action={self.action_description}"
        return result


class ActionResultBuilder:
    """Builder for ActionResult. Thread-safe construction.

    This builder allows mutable construction of results, then creates
    an immutable ActionResult when build() is called.

    Thread Safety:
        Protected by internal RLock for thread-safe construction.
        Multiple threads can add data concurrently.

    Example:
        result = (ActionResultBuilder()
                 .add_match(match1)
                 .add_match(match2)
                 .with_success(True)
                 .with_description("Found 2 elements")
                 .build())
    """

    def __init__(self, action_config: ActionConfig | None = None) -> None:
        """Initialize builder with optional configuration.

        Args:
            action_config: Configuration to associate with the result
        """
        self._lock = threading.RLock()
        self._action_config = action_config
        self._success = False
        self._matches: list[Match] = []
        self._times_acted_on = 0
        self._text_parts: list[str] = []
        self._defined_regions: list[Region] = []
        self._movements: list[Movement] = []
        self._execution_history: list[ExecutionRecord] = []
        self._active_states: set[str] = set()
        self._action_description = ""
        self._output_text = ""
        self._selected_text = ""
        self._duration: timedelta | None = None
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None

    def with_success(self, success: bool) -> ActionResultBuilder:
        """Set success status.

        Args:
            success: Whether action succeeded

        Returns:
            Self for method chaining
        """
        with self._lock:
            self._success = success
        return self

    def add_match(self, match: Match) -> ActionResultBuilder:
        """Add a match to the result.

        Thread-safe: Can be called from multiple threads.

        Args:
            match: Match to add

        Returns:
            Self for method chaining
        """
        with self._lock:
            self._matches.append(match)
            # Extract state information if available
            if hasattr(match, "get_state_object_data"):
                state_data = match.get_state_object_data()
                if state_data and hasattr(state_data, "get_owner_state_name"):
                    self._active_states.add(state_data.get_owner_state_name())
        return self

    def set_times_acted_on(self, times: int) -> ActionResultBuilder:
        """Set the times acted on counter.

        Thread-safe: Can be called from multiple threads.

        Args:
            times: The count to set

        Returns:
            Self for method chaining
        """
        with self._lock:
            self._times_acted_on = times
            # Update all matches
            for match in self._matches:
                if hasattr(match, "set_times_acted_on"):
                    match.set_times_acted_on(times)
        return self

    def add_text(self, text: str) -> ActionResultBuilder:
        """Add text to the accumulated text.

        Thread-safe: Can be called from multiple threads.

        Args:
            text: Text to add

        Returns:
            Self for method chaining
        """
        with self._lock:
            self._text_parts.append(text)
        return self

    def add_defined_region(self, region: Region) -> ActionResultBuilder:
        """Add a defined region.

        Thread-safe: Can be called from multiple threads.

        Args:
            region: Region to add

        Returns:
            Self for method chaining
        """
        with self._lock:
            self._defined_regions.append(region)
        return self

    def add_movement(self, movement: Movement) -> ActionResultBuilder:
        """Add a movement.

        Thread-safe: Can be called from multiple threads.

        Args:
            movement: Movement to add

        Returns:
            Self for method chaining
        """
        with self._lock:
            self._movements.append(movement)
        return self

    def add_execution_record(self, record: ExecutionRecord) -> ActionResultBuilder:
        """Add an execution record.

        Thread-safe: Can be called from multiple threads.

        Args:
            record: Execution record to add

        Returns:
            Self for method chaining
        """
        with self._lock:
            self._execution_history.append(record)
        return self

    def with_description(self, description: str) -> ActionResultBuilder:
        """Set action description.

        Args:
            description: Human-readable action description

        Returns:
            Self for method chaining
        """
        with self._lock:
            self._action_description = description
        return self

    def with_output_text(self, text: str) -> ActionResultBuilder:
        """Set output text.

        Args:
            text: Formatted output text

        Returns:
            Self for method chaining
        """
        with self._lock:
            self._output_text = text
        return self

    def with_selected_text(self, text: str) -> ActionResultBuilder:
        """Set selected text.

        Args:
            text: Selected/highlighted text

        Returns:
            Self for method chaining
        """
        with self._lock:
            self._selected_text = text
        return self

    def with_timing(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        duration: timedelta | None = None,
    ) -> ActionResultBuilder:
        """Set timing information.

        Args:
            start: Start timestamp
            end: End timestamp
            duration: Total duration

        Returns:
            Self for method chaining
        """
        with self._lock:
            self._start_time = start
            self._end_time = end
            self._duration = duration
        return self

    def build(self) -> ActionResult:
        """Build immutable ActionResult.

        Thread-safe: Creates immutable result from current state.

        Returns:
            Immutable ActionResult
        """
        with self._lock:
            # Infer success from matches if not explicitly set
            success = self._success or len(self._matches) > 0

            # Combine text parts
            text = "".join(self._text_parts)

            return ActionResult(
                success=success,
                matches=tuple(self._matches),
                times_acted_on=self._times_acted_on,
                text=text,
                defined_regions=tuple(self._defined_regions),
                movements=tuple(self._movements),
                execution_history=tuple(self._execution_history),
                active_states=frozenset(self._active_states),
                action_description=self._action_description,
                output_text=self._output_text,
                selected_text=self._selected_text,
                duration=self._duration,
                start_time=self._start_time,
                end_time=self._end_time,
                action_config=self._action_config,
            )


# Forward references for type hints
class Movement:
    """Placeholder for Movement class."""

    pass


class ExecutionRecord:
    """Placeholder for ExecutionRecord class."""

    pass
