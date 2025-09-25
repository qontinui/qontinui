"""Action result - ported from Qontinui framework.

Comprehensive results container for action executions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .object_collection import ObjectCollection


@dataclass
class ActionResult:
    """Comprehensive results container for all action executions.

    Port of ActionResult from Qontinui framework class (simplified version).

    ActionResult serves as the universal return type for all actions, encapsulating not just
    pattern matching results but all information generated during action execution. This
    unified approach simplifies the API and provides consistent access to action outcomes
    regardless of the action type.
    """

    action_description: str = ""
    """Human-readable description of the action performed."""

    success: bool = False
    """Indicates whether the action achieved its intended goal."""

    output_text: str = ""
    """Formatted text output for reporting and logging."""

    match_list: list["Match"] = field(default_factory=list)
    """List of all matches found during action execution."""

    initial_match_list: list["Match"] = field(default_factory=list)
    """Initial matches before any filtering or processing."""

    max_matches: int = -1
    """Maximum number of matches to return (-1 for unlimited)."""

    text: Optional["Text"] = None
    """Accumulated text content from all matches."""

    selected_text: str = ""
    """Specific text selected or highlighted during the action."""

    active_states: set[str] = field(default_factory=set)
    """Names of states identified as active during action execution."""

    duration: timedelta | None = None
    """Total time taken for action execution."""

    start_time: datetime | None = None
    """Timestamp when action execution began."""

    end_time: datetime | None = None
    """Timestamp when action execution completed."""

    defined_regions: list["Region"] = field(default_factory=list)
    """Regions created or captured by DEFINE actions."""

    movements: list["Movement"] = field(default_factory=list)
    """List of movements performed during action execution."""

    execution_history: list["ActionRecord"] = field(default_factory=list)
    """Ordered history of action execution steps."""

    times_acted_on: int = 0
    """Count of how many times this object was acted upon."""

    action_config: Optional["ActionConfig"] = None
    """Configuration used for this action execution."""

    def __init__(self, action_config: Optional["ActionConfig"] = None):
        """Initialize ActionResult with optional configuration.

        Args:
            action_config: Configuration that will control the action execution
        """
        self.action_config = action_config
        self.action_description = ""
        self.success = False
        self.output_text = ""
        self.match_list = []
        self.initial_match_list = []
        self.max_matches = -1
        self.text = None
        self.selected_text = ""
        self.active_states = set()
        self.duration = None
        self.start_time = None
        self.end_time = None
        self.defined_regions = []
        self.movements = []
        self.execution_history = []
        self.times_acted_on = 0

    def add(self, *matches: "Match") -> None:
        """Add one or more matches to the result set.

        Also extracts and records any state information from the matches.

        Args:
            matches: Variable number of Match objects to add
        """
        for match in matches:
            self.match_list.append(match)
            # Extract state information if available
            if hasattr(match, "get_state_object_data"):
                state_data = match.get_state_object_data()
                if state_data and hasattr(state_data, "get_owner_state_name"):
                    self.active_states.add(state_data.get_owner_state_name())

    def get_match_list(self) -> list["Match"]:
        """Get the list of all matches found during action execution.

        Returns:
            List of matches
        """
        return self.match_list

    def set_match_list(self, matches: list["Match"]) -> None:
        """Set the match list directly.

        Args:
            matches: List of matches to set
        """
        self.match_list = matches if matches else []

    def get_best_match(self) -> Optional["Match"]:
        """Find the match with the highest similarity score.

        Returns:
            Optional containing the best match, or None if no matches
        """
        if not self.match_list:
            return None
        return max(self.match_list, key=lambda m: m.get_score() if hasattr(m, "get_score") else 0)

    def get_best_location(self) -> Optional["Location"]:
        """Get the target location of the best scoring match.

        Returns:
            Optional containing the location, or None if no matches
        """
        best = self.get_best_match()
        if best and hasattr(best, "get_target"):
            return best.get_target()
        return None

    def size(self) -> int:
        """Get the number of matches found.

        Returns:
            Count of matches in the result
        """
        return len(self.match_list)

    def is_empty(self) -> bool:
        """Check if the action found any matches.

        Returns:
            True if no matches were found
        """
        return len(self.match_list) == 0

    def set_times_acted_on(self, times: int) -> None:
        """Update the action count for all matches.

        Args:
            times: The count to set for all matches
        """
        self.times_acted_on = times
        for match in self.match_list:
            if hasattr(match, "set_times_acted_on"):
                match.set_times_acted_on(times)

    def add_string(self, text: str) -> None:
        """Add a text string to the accumulated text results.

        Args:
            text: Text to add to the results
        """
        if self.text is None:
            from ..model.element import Text

            self.text = Text()
        if hasattr(self.text, "add"):
            self.text.add(text)

    def add_defined_region(self, region: "Region") -> None:
        """Add a region to the defined regions collection.

        Args:
            region: The region to add
        """
        self.defined_regions.append(region)

    def get_defined_region(self) -> Optional["Region"]:
        """Get the primary region defined by this action.

        Returns:
            The first defined region or None
        """
        if self.defined_regions:
            return self.defined_regions[0]
        return None

    def add_movement(self, movement: "Movement") -> None:
        """Add a movement to the result.

        Args:
            movement: The movement to add
        """
        self.movements.append(movement)

    def get_movement(self) -> Optional["Movement"]:
        """Return an Optional containing the first movement from the action.

        Returns:
            An Optional containing the first Movement if one exists
        """
        if self.movements:
            return self.movements[0]
        return None

    def add_execution_record(self, record: "ActionRecord") -> None:
        """Add an action record to the execution history.

        Args:
            record: The action record to add
        """
        self.execution_history.append(record)

    def add_match_objects(self, matches: "ActionResult") -> None:
        """Merge match objects from another ActionResult.

        Args:
            matches: Source ActionResult containing matches to add
        """
        if matches:
            for match in matches.get_match_list():
                self.add(match)

    def add_all_results(self, matches: "ActionResult") -> None:
        """Merge all data from another ActionResult.

        Args:
            matches: Source ActionResult to merge completely
        """
        if matches:
            self.add_match_objects(matches)
            self.add_non_match_results(matches)

    def add_non_match_results(self, matches: "ActionResult") -> None:
        """Merge non-match data from another ActionResult.

        Args:
            matches: Source ActionResult containing data to merge
        """
        if matches:
            if matches.text:
                self.text = matches.text
            if matches.selected_text:
                self.selected_text = matches.selected_text
            self.active_states.update(matches.active_states)
            self.defined_regions.extend(matches.defined_regions)
            self.movements.extend(matches.movements)
            self.execution_history.extend(matches.execution_history)

    def as_object_collection(self) -> "ObjectCollection":
        """Convert this result into an ObjectCollection.

        Returns:
            New ObjectCollection containing these results
        """
        from .object_collection import ObjectCollectionBuilder

        return ObjectCollectionBuilder().with_matches(self).build()

    def print(self) -> None:
        """Print all matches to standard output."""
        for match in self.match_list:
            print(match)

    def get_success_symbol(self) -> str:
        """Get a visual symbol representing action success or failure.

        Returns:
            Unicode symbol indicating success (✓) or failure (✗)
        """
        return "✓" if self.success else "✗"

    def get_summary(self) -> str:
        """Get a summary of the action result.

        Returns:
            Summary string
        """
        summary = []
        if self.action_config:
            summary.append(f"Action: {self.action_config.__class__.__name__}")
        summary.append(f"Success: {self.success}")
        summary.append(f"Number of matches: {self.size()}")
        if self.active_states:
            summary.append(f"Active states: {', '.join(self.active_states)}")
        if self.text:
            summary.append(f"Extracted text: {self.text}")
        return "\n".join(summary)

    def __str__(self) -> str:
        """String representation for debugging."""
        result = f"ActionResult: size={self.size()}"
        for match in self.match_list:
            result += f" {match}"
        return result


# Forward references
class Match:
    """Placeholder for Match class."""

    pass


class Text:
    """Placeholder for Text class."""

    pass


class Region:
    """Placeholder for Region class."""

    pass


class Location:
    """Placeholder for Location class."""

    pass


class Movement:
    """Placeholder for Movement class."""

    pass


class ActionRecord:
    """Placeholder for ActionRecord class."""

    pass


class ActionConfig:
    """Placeholder for ActionConfig class."""

    pass
