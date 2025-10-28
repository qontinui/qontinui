"""Click action builder for action chains.

Provides a wrapper for Click actions in chains.
"""

from typing import Any

from ....basic.click.click import Click


class ClickBuilder:
    """Builder for Click actions in chains.

    Wraps a Click action to be used in action chains.
    """

    def __init__(self) -> None:
        """Initialize click builder."""
        self.action = Click()

    def execute(self, target: Any) -> bool:
        """Execute click action on target.

        Args:
            target: The target to click

        Returns:
            True if click succeeded
        """
        return self.action.execute(target)
