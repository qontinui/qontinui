"""
Majority merge strategy implementation.

This strategy executes when a majority (or custom threshold) of inputs
have arrived.
"""

from .merge_base import MergeStrategy


class MajorityStrategy(MergeStrategy):
    """
    Execute when a majority of inputs have arrived.

    The merge executes when more than half of the expected inputs have arrived.

    Use cases:
    - Consensus-based workflows
    - Quorum requirements
    - Voting mechanisms
    - Fault tolerance (execute when majority succeeds)
    """

    def __init__(self, threshold: float | None = None) -> None:
        """
        Initialize MajorityStrategy.

        Args:
            threshold: Threshold as fraction (0.5 = 50%, 0.67 = 67%, etc.)
                      Default is 0.5 (simple majority)
        """
        self.threshold = threshold or 0.5
        if not 0 < self.threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        super().__init__(
            name="majority", description=f"Execute when {self.threshold*100:.0f}% of inputs arrive"
        )

    def should_execute(self, received_inputs: int, total_inputs: int, **kwargs) -> bool:
        """Execute when threshold percentage of inputs received."""
        if total_inputs == 0:
            return False

        required = int(total_inputs * self.threshold)
        # At least one input required
        required = max(1, required)

        return received_inputs >= required

    def get_merge_mode(self) -> str:
        """Use all inputs that have arrived."""
        return "partial"
