"""
Wait-first merge strategy implementation.

This strategy executes on the first input to arrive and ignores all others.
"""

from .merge_base import MergeStrategy


class WaitFirstStrategy(MergeStrategy):
    """
    Execute on the first input to arrive, ignore all others.

    Similar to WaitAnyStrategy but explicitly designed to execute only once
    and ignore all subsequent inputs.

    Use cases:
    - Race conditions where first to finish wins
    - Timeout scenarios with a fast path and slow path
    - Competitive execution where first success is sufficient

    Difference from WaitAnyStrategy:
    - WaitFirstStrategy explicitly discards late arrivals
    - WaitAnyStrategy may process late arrivals depending on configuration
    """

    def __init__(self) -> None:
        super().__init__(name="wait_first", description="Execute on first input, ignore all others")
        self._first_received: str | None = None
        self._executed = False

    def should_execute(
        self,
        received_inputs: int,
        total_inputs: int,
        first_action_id: str | None = None,
        **kwargs,
    ) -> bool:
        """Execute only on the first input."""
        if received_inputs == 0:
            return False

        if self._executed:
            return False

        # If this is the first input, prepare to execute
        if received_inputs == 1 and first_action_id:
            self._first_received = first_action_id
            return True

        return False

    def get_merge_mode(self) -> str:
        """Use only the first input."""
        return "first"

    def get_first_action_id(self) -> str | None:
        """Get the ID of the first action that provided input."""
        return self._first_received

    def mark_executed(self) -> None:
        """Mark that execution has occurred."""
        self._executed = True

    def reset(self) -> None:
        """Reset execution state."""
        self._first_received = None
        self._executed = False
