"""Type action builder for action chains.

Provides a wrapper for Type actions in chains.
"""

from typing import Any

from ....basic.type.type_action import TypeAction


class TypeBuilder:
    """Builder for Type actions in chains.

    Wraps a TypeAction to handle text input in action chains.
    """

    def __init__(self, text: str) -> None:
        """Initialize type builder.

        Args:
            text: The text to type
        """
        self.text = text
        self.action = TypeAction()

    def execute(self, target: Any | None = None) -> bool:
        """Execute type action with text.

        Args:
            target: Optional target element to type into

        Returns:
            True if type action succeeded
        """
        return self.action.execute(self.text, target)
