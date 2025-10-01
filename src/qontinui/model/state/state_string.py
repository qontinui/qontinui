"""StateString - ported from Qontinui framework.

Strings associated with states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qontinui.actions.action_result import ActionResult
    from qontinui.model.match.match import Match
    from qontinui.model.state.state import State


@dataclass
class StateString:
    """String associated with a state.

    Port of StateString from Qontinui framework class.
    Represents text that appears in or identifies a state.
    """

    string: str
    name: str | None = None
    owner_state: State | None = None

    # String properties
    _identifier: bool = False  # If true, used to identify state
    _input_text: bool = False  # If true, used as input text
    _expected_text: bool = False  # If true, expected to appear in state
    _regex: bool = False  # If true, string is a regex pattern

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize string name."""
        if self.name is None:
            # Use first few chars of string as name
            self.name = self.string[:20] if len(self.string) > 20 else self.string

    def find_on_screen(self) -> list[Match]:
        """Find this string on screen using OCR.

        Returns:
            List of matches where text was found
        """
        # This would use OCR to find text
        # For now, return empty list as placeholder
        return []

    def exists_on_screen(self) -> bool:
        """Check if string exists on screen.

        Returns:
            True if string is found
        """
        matches = self.find_on_screen()
        return len(matches) > 0

    def get_string(self) -> str:
        """Get the string value.

        Returns:
            The string value
        """
        return self.string

    def type(self) -> ActionResult:
        """Type this string.

        Returns:
            ActionResult from typing
        """
        from ..actions import Action, TypeOptions

        action = Action(TypeOptions())
        result: ActionResult = action.type_text(self.string)
        return result

    def matches(self, text: str) -> bool:
        """Check if text matches this string.

        Args:
            text: Text to check

        Returns:
            True if matches
        """
        if self._regex:
            import re

            return bool(re.match(self.string, text))
        else:
            return self.string == text

    def contains(self, text: str) -> bool:
        """Check if text contains this string.

        Args:
            text: Text to check

        Returns:
            True if contains
        """
        if self._regex:
            import re

            return bool(re.search(self.string, text))
        else:
            return self.string in text

    def set_identifier(self, identifier: bool = True) -> StateString:
        """Set whether this identifies the state (fluent).

        Args:
            identifier: True if identifier

        Returns:
            Self for chaining
        """
        self._identifier = identifier
        return self

    def set_input_text(self, input_text: bool = True) -> StateString:
        """Set whether this is input text (fluent).

        Args:
            input_text: True if input text

        Returns:
            Self for chaining
        """
        self._input_text = input_text
        return self

    def set_expected_text(self, expected: bool = True) -> StateString:
        """Set whether this is expected text (fluent).

        Args:
            expected: True if expected

        Returns:
            Self for chaining
        """
        self._expected_text = expected
        return self

    def set_regex(self, regex: bool = True) -> StateString:
        """Set whether this is a regex pattern (fluent).

        Args:
            regex: True if regex

        Returns:
            Self for chaining
        """
        self._regex = regex
        return self

    @property
    def is_identifier(self) -> bool:
        """Check if this identifies the state."""
        return self._identifier

    @property
    def is_input_text(self) -> bool:
        """Check if this is input text."""
        return self._input_text

    @property
    def is_expected_text(self) -> bool:
        """Check if this is expected text."""
        return self._expected_text

    @property
    def is_regex(self) -> bool:
        """Check if this is a regex pattern."""
        return self._regex

    def get_owner_state_name(self) -> str:
        """Get the owner state name.

        Returns:
            Owner state name or empty string if no owner
        """
        if self.owner_state is None:
            return ""
        return self.owner_state.name if hasattr(self.owner_state, "name") else ""

    def set_times_acted_on(self, times: int) -> None:
        """Set times acted on count.

        Args:
            times: Number of times acted on
        """
        # StateString doesn't track action history like other state objects
        # This is a placeholder for API compatibility
        pass

    def __str__(self) -> str:
        """String representation."""
        state_name = self.owner_state.name if self.owner_state else "None"
        return f"StateString('{self.name}' in state '{state_name}')"
