"""Builder pattern for ActionResult construction.

Provides fluent interface for creating and configuring ActionResult instances
with method chaining.
"""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..find.match import Match
    from ..model.element.region import Region
    from .action_config import ActionConfig
    from .action_result import ActionResult


class ActionResultBuilder:
    """Builder for constructing ActionResult instances with fluent interface.

    Provides method chaining for configuring result properties before building.
    Separates construction logic from the data class itself.

    Example:
        result = (ActionResultBuilder()
                 .with_success(True)
                 .with_matches([match1, match2])
                 .with_description("Found 2 elements")
                 .build())
    """

    def __init__(self, action_config: Optional["ActionConfig"] = None) -> None:
        """Initialize builder with optional configuration.

        Args:
            action_config: Configuration to associate with the result
        """
        self._config = action_config
        self._success = False
        self._description = ""
        self._output_text = ""
        self._matches: list[Match] = []
        self._initial_matches: list[Match] = []
        self._max_matches = -1
        self._text: Text | None = None
        self._selected_text = ""
        self._active_states: set[str] = set()
        self._duration: timedelta | None = None
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None
        self._defined_regions: list[Region] = []
        self._movements: list[Movement] = []
        self._execution_history: list[ActionRecord] = []
        self._times_acted_on = 0

    def with_success(self, success: bool) -> "ActionResultBuilder":
        """Set success status.

        Args:
            success: Whether action succeeded

        Returns:
            Self for method chaining
        """
        self._success = success
        return self

    def with_description(self, description: str) -> "ActionResultBuilder":
        """Set action description.

        Args:
            description: Human-readable action description

        Returns:
            Self for method chaining
        """
        self._description = description
        return self

    def with_output_text(self, text: str) -> "ActionResultBuilder":
        """Set output text.

        Args:
            text: Formatted output text

        Returns:
            Self for method chaining
        """
        self._output_text = text
        return self

    def with_matches(self, matches: list["Match"]) -> "ActionResultBuilder":
        """Set match list.

        Args:
            matches: List of matches to set

        Returns:
            Self for method chaining
        """
        self._matches = matches if matches else []
        return self

    def add_match(self, match: "Match") -> "ActionResultBuilder":
        """Add a single match.

        Args:
            match: Match to add

        Returns:
            Self for method chaining
        """
        self._matches.append(match)
        # Extract state information if available
        if hasattr(match, "get_state_object_data"):
            state_data = match.get_state_object_data()
            if state_data and hasattr(state_data, "get_owner_state_name"):
                self._active_states.add(state_data.get_owner_state_name())
        return self

    def with_initial_matches(self, matches: list["Match"]) -> "ActionResultBuilder":
        """Set initial matches before filtering.

        Args:
            matches: Initial match list

        Returns:
            Self for method chaining
        """
        self._initial_matches = matches if matches else []
        return self

    def with_max_matches(self, max_matches: int) -> "ActionResultBuilder":
        """Set maximum number of matches.

        Args:
            max_matches: Maximum matches (-1 for unlimited)

        Returns:
            Self for method chaining
        """
        self._max_matches = max_matches
        return self

    def with_text(self, text: "Text") -> "ActionResultBuilder":
        """Set accumulated text content.

        Args:
            text: Text content

        Returns:
            Self for method chaining
        """
        self._text = text
        return self

    def with_selected_text(self, text: str) -> "ActionResultBuilder":
        """Set selected text.

        Args:
            text: Selected/highlighted text

        Returns:
            Self for method chaining
        """
        self._selected_text = text
        return self

    def with_active_states(self, states: set[str]) -> "ActionResultBuilder":
        """Set active states.

        Args:
            states: Set of active state names

        Returns:
            Self for method chaining
        """
        self._active_states = states
        return self

    def with_timing(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        duration: timedelta | None = None,
    ) -> "ActionResultBuilder":
        """Set timing information.

        Args:
            start: Start timestamp
            end: End timestamp
            duration: Total duration

        Returns:
            Self for method chaining
        """
        self._start_time = start
        self._end_time = end
        self._duration = duration
        return self

    def with_defined_regions(self, regions: list["Region"]) -> "ActionResultBuilder":
        """Set defined regions.

        Args:
            regions: List of regions

        Returns:
            Self for method chaining
        """
        self._defined_regions = regions
        return self

    def add_defined_region(self, region: "Region") -> "ActionResultBuilder":
        """Add a single defined region.

        Args:
            region: Region to add

        Returns:
            Self for method chaining
        """
        self._defined_regions.append(region)
        return self

    def with_movements(self, movements: list["Movement"]) -> "ActionResultBuilder":
        """Set movements.

        Args:
            movements: List of movements

        Returns:
            Self for method chaining
        """
        self._movements = movements
        return self

    def add_movement(self, movement: "Movement") -> "ActionResultBuilder":
        """Add a single movement.

        Args:
            movement: Movement to add

        Returns:
            Self for method chaining
        """
        self._movements.append(movement)
        return self

    def with_execution_history(
        self, history: list["ActionRecord"]
    ) -> "ActionResultBuilder":
        """Set execution history.

        Args:
            history: List of action records

        Returns:
            Self for method chaining
        """
        self._execution_history = history
        return self

    def add_execution_record(self, record: "ActionRecord") -> "ActionResultBuilder":
        """Add a single execution record.

        Args:
            record: Action record to add

        Returns:
            Self for method chaining
        """
        self._execution_history.append(record)
        return self

    def with_times_acted_on(self, times: int) -> "ActionResultBuilder":
        """Set times acted on counter.

        Args:
            times: Action count

        Returns:
            Self for method chaining
        """
        self._times_acted_on = times
        return self

    def build(self) -> "ActionResult":
        """Build the final ActionResult instance.

        Returns:
            Configured ActionResult instance
        """
        from .action_result import ActionResult

        # Convert builder fields to ActionResult constructor arguments
        # ActionResult is a frozen dataclass, so we must pass all arguments at construction
        text_str = str(self._text) if self._text is not None else ""

        result = ActionResult(
            success=self._success,
            matches=tuple(self._matches),
            times_acted_on=self._times_acted_on,
            text=text_str,
            defined_regions=tuple(self._defined_regions),
            movements=tuple(self._movements),  # type: ignore[arg-type]
            execution_history=tuple(self._execution_history),  # type: ignore[arg-type]
            active_states=frozenset(self._active_states),
        )
        return result


# Forward references
class Text:
    """Placeholder for Text class."""

    pass


class Movement:
    """Placeholder for Movement class."""

    pass


class ActionRecord:
    """Placeholder for ActionRecord class."""

    pass
