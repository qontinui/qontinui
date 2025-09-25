"""ActionRecord class - ported from Qontinui framework.

Records match results and context at a specific point in time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...actions.action_config import ActionConfig
    from ..match.match import Match


@dataclass
class ActionRecord:
    """Records match results and context at a specific point in time.

    Port of ActionRecord from Qontinui framework class.

    ActionRecord captures comprehensive information about a match operation, including the
    matches found (or not found), the action options used, timing data, and contextual state
    information. These records form the foundation of learning and mocking capabilities,
    enabling the framework to simulate realistic GUI behavior based on historical data.

    Key components captured:
    - Match Results: List of matches found (empty for failed searches)
    - Action Context: The ActionConfig that produced this result
    - Timing Data: Duration of the operation for performance analysis
    - State Context: Which state the match occurred in
    - Success Indicators: Both action success and result success flags
    - Text Data: Extracted text for text-based operations

    Snapshot creation strategy:
    - One record per action completion, not per find operation
    - Avoids skewing data with repeated searches in wait loops
    - Captures representative examples of real-world behavior

    Mock operation support:
    - Provides realistic match/failure distributions based on history
    - Action-specific records prevent cross-action interference
    - Enables accurate simulation of Find.ALL and Find.EACH operations
    - Supports extraction of best matches for Find.FIRST/BEST simulation
    """

    action_config: ActionConfig | None = None
    match_list: list[Match] = field(default_factory=list)
    text: str = ""
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    action_success: bool = False  # Action was successfully performed
    result_success: bool = False  # Result was successful (app-defined)
    state_name: str = "NULL"  # State where match occurred
    state_id: int = -1  # State ID for database reference

    def was_found(self) -> bool:
        """Check if anything was found.

        Returns:
            True if matches or text were found
        """
        return len(self.match_list) > 0 or len(self.text) > 0

    def add_match(self, match: Match) -> None:
        """Add a match to the record.

        Args:
            match: Match to add
        """
        self.match_list.append(match)

    def add_match_list(self, matches: list[Match]) -> None:
        """Add multiple matches to the record.

        Args:
            matches: List of matches to add
        """
        self.match_list.extend(matches)

    def set_string(self, text: str) -> None:
        """Set the text result.

        Args:
            text: Text extracted from action
        """
        self.text = text

    def has_same_results_as(self, other: ActionRecord) -> bool:
        """Check if two records have the same results.

        Args:
            other: Another ActionRecord to compare

        Returns:
            True if match lists and text are the same
        """
        # Check if all matches in self exist in other
        for match in self.match_list:
            if not any(m == match for m in other.match_list):
                return False

        # Check if all matches in other exist in self
        for match in other.match_list:
            if not any(m == match for m in self.match_list):
                return False

        # Check text equality
        return self.text == other.text

    def get_action_type(self) -> str:
        """Get action type from config.

        Returns:
            String representation of action type
        """
        if not self.action_config:
            return "UNKNOWN"

        class_name = self.action_config.__class__.__name__

        # Map config class names to action types
        if "Click" in class_name:
            return "CLICK"
        elif "Find" in class_name or "Pattern" in class_name:
            return "FIND"
        elif "Type" in class_name:
            return "TYPE"
        elif "Drag" in class_name:
            return "DRAG"
        elif "Move" in class_name or "Mouse" in class_name:
            return "MOVE"
        elif "Highlight" in class_name:
            return "HIGHLIGHT"
        elif "Scroll" in class_name:
            return "SCROLL"
        elif "KeyDown" in class_name:
            return "KEY_DOWN"
        elif "KeyUp" in class_name:
            return "KEY_UP"
        else:
            return class_name.replace("Options", "").upper()

    def print(self) -> None:
        """Print record details."""
        print(f"{self.timestamp.strftime('%m-%d %H:%M:%S')}", end="")

        if self.action_config:
            print(f" {self.get_action_type()}", end="")

        for match in self.match_list:
            print(
                f" {match.region.x},{match.region.y},{match.region.width},{match.region.height}",
                end="",
            )

        if self.text:
            print(f" {self.text}", end="")

        print()  # New line

    def __str__(self) -> str:
        """String representation."""
        return (
            f"ActionRecord(action={self.get_action_type()}, "
            f"matches={len(self.match_list)}, "
            f"success={self.action_success}, "
            f"duration={self.duration:.3f}s)"
        )

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"ActionRecord(action_config={self.action_config}, "
            f"match_list={self.match_list}, "
            f"text='{self.text}', "
            f"duration={self.duration}, "
            f"timestamp={self.timestamp}, "
            f"action_success={self.action_success}, "
            f"result_success={self.result_success}, "
            f"state_name='{self.state_name}', "
            f"state_id={self.state_id})"
        )

    @classmethod
    def builder(cls) -> ActionRecordBuilder:
        """Create a new ActionRecordBuilder.

        Returns:
            New builder instance
        """
        return ActionRecordBuilder()


class ActionRecordBuilder:
    """Builder for ActionRecord class."""

    def __init__(self):
        """Initialize builder with defaults."""
        self.action_config = None
        self.match_list = []
        self.text = ""
        self.duration = 0.0
        self.timestamp = None
        self.action_success = False
        self.result_success = False
        self.state_name = "NULL"
        self.state_id = -1

    def set_action_config(self, config: ActionConfig) -> ActionRecordBuilder:
        """Set action config (fluent).

        Args:
            config: ActionConfig used

        Returns:
            Self for chaining
        """
        self.action_config = config
        return self

    def set_match_list(self, matches: list[Match]) -> ActionRecordBuilder:
        """Set match list (fluent).

        Args:
            matches: List of matches found

        Returns:
            Self for chaining
        """
        self.match_list = matches
        return self

    def add_match(self, match: Match) -> ActionRecordBuilder:
        """Add a match (fluent).

        Args:
            match: Match to add

        Returns:
            Self for chaining
        """
        self.match_list.append(match)
        return self

    def set_text(self, text: str) -> ActionRecordBuilder:
        """Set text result (fluent).

        Args:
            text: Text extracted

        Returns:
            Self for chaining
        """
        self.text = text
        return self

    def set_duration(self, duration: float) -> ActionRecordBuilder:
        """Set operation duration (fluent).

        Args:
            duration: Duration in seconds

        Returns:
            Self for chaining
        """
        self.duration = duration
        return self

    def set_action_success(self, success: bool) -> ActionRecordBuilder:
        """Set action success (fluent).

        Args:
            success: Whether action succeeded

        Returns:
            Self for chaining
        """
        self.action_success = success
        return self

    def set_result_success(self, success: bool) -> ActionRecordBuilder:
        """Set result success (fluent).

        Args:
            success: Whether result was successful

        Returns:
            Self for chaining
        """
        self.result_success = success
        return self

    def set_state(self, state_name: str, state_id: int = -1) -> ActionRecordBuilder:
        """Set state context (fluent).

        Args:
            state_name: Name of state
            state_id: Optional state ID

        Returns:
            Self for chaining
        """
        self.state_name = state_name
        self.state_id = state_id
        return self

    def build(self) -> ActionRecord:
        """Build the ActionRecord.

        Returns:
            Configured ActionRecord instance
        """
        return ActionRecord(
            action_config=self.action_config,
            match_list=self.match_list.copy(),
            text=self.text,
            duration=self.duration,
            timestamp=self.timestamp or datetime.now(),
            action_success=self.action_success,
            result_success=self.result_success,
            state_name=self.state_name,
            state_id=self.state_id,
        )
