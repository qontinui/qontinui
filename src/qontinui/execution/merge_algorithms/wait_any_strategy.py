"""
Wait-any merge strategy implementation.

This strategy executes when any incoming path arrives.
"""

from .merge_base import MergeStrategy


class WaitAnyStrategy(MergeStrategy):
    """
    Execute when any incoming path arrives.

    The merge node executes as soon as any input arrives, without waiting
    for other paths. Subsequent inputs may trigger re-execution if configured.

    Use cases:
    - First-response-wins scenarios
    - Optional parallel branches (any one succeeding is sufficient)
    - Fallback mechanisms where multiple paths are tried

    Note: This can lead to multiple executions if not carefully managed.
    """

    def __init__(self, allow_multiple_executions: bool = False) -> None:
        """
        Initialize WaitAnyStrategy.

        Args:
            allow_multiple_executions: If True, execute on each arriving input.
                                      If False, execute only once on first input.
        """
        super().__init__(name="wait_any", description="Execute as soon as any input arrives")
        self.allow_multiple_executions = allow_multiple_executions
        self._executed = False

    def should_execute(self, received_inputs: int, total_inputs: int, **kwargs) -> bool:
        """Execute when any input is received."""
        if received_inputs == 0:
            return False

        if not self.allow_multiple_executions and self._executed:
            return False

        return True

    def get_merge_mode(self) -> str:
        """Use only the inputs that have arrived."""
        return "any"

    def mark_executed(self) -> None:
        """Mark that execution has occurred."""
        self._executed = True

    def reset(self) -> None:
        """Reset execution state."""
        self._executed = False
