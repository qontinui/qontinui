"""String collection builder for ObjectCollection.

Handles StateStrings and text.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...model.state import StateString


class StringCollectionBuilder:
    """Builder for string-related objects in ObjectCollection.

    Handles:
    - StateStrings
    - Plain strings (converted to StateStrings)
    """

    def __init__(self) -> None:
        """Initialize builder with empty list."""
        self.state_strings: list["StateString"] = []

    def with_strings(self, *strings) -> "StringCollectionBuilder":
        """Add strings to collection.

        Args:
            strings: Variable number of string or StateString objects

        Returns:
            This builder for method chaining
        """
        from ...model.state.state_string import StateString

        for string in strings:
            if isinstance(string, str):
                self.state_strings.append(StateString(string))
            elif isinstance(string, StateString):
                self.state_strings.append(string)
        return self

    def set_strings(self, strings: list["StateString"]) -> "StringCollectionBuilder":
        """Set strings list.

        Args:
            strings: List of StateString objects

        Returns:
            This builder for method chaining
        """
        self.state_strings = strings
        return self

    def build(self) -> list["StateString"]:
        """Build and return the state strings list.

        Returns:
            Copy of state strings list
        """
        return self.state_strings.copy()
