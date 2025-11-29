"""Drag action builder for action chains.

Provides a wrapper for Drag actions in chains.
"""

from typing import Any

from ...drag.drag import Drag


class DragBuilder:
    """Builder for Drag actions in chains.

    Wraps a Drag action to handle start and end points in action chains.
    """

    def __init__(self, start: Any, end: Any) -> None:
        """Initialize drag builder.

        Args:
            start: The drag start position
            end: The drag end position
        """
        self.start = start
        self.end = end
        self.action = Drag()

    def execute(self) -> bool:
        """Execute drag action.

        Returns:
            True if drag action succeeded
        """
        return self.action.execute(self.start, self.end)  # type: ignore[no-any-return, attr-defined]
