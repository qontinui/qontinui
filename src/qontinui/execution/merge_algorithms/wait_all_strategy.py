"""
Wait-all merge strategy implementation.

This strategy waits for all incoming paths before executing.
"""

from .merge_base import MergeStrategy


class WaitAllStrategy(MergeStrategy):
    """
    Wait for all incoming paths before executing.

    This is the default and most common strategy. The merge node only executes
    when all expected inputs have arrived.

    Use cases:
    - Synchronization points in parallel workflows
    - Combining results from multiple parallel branches
    - Ensuring all prerequisites are met before continuing
    """

    def __init__(self) -> None:
        super().__init__(name="wait_all", description="Wait for all incoming paths to complete")

    def should_execute(self, received_inputs: int, total_inputs: int, **kwargs) -> bool:
        """Execute only when all inputs received."""
        return received_inputs >= total_inputs

    def get_merge_mode(self) -> str:
        """Merge all input contexts."""
        return "all"
