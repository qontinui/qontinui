"""
Timeout merge strategy implementation.

This strategy waits for inputs with a timeout, executing with whatever
is available when the timeout expires.
"""

from datetime import datetime

from .merge_base import MergeStrategy


class TimeoutStrategy(MergeStrategy):
    """
    Wait for inputs with a timeout, execute with whatever is available.

    This strategy waits for all inputs up to a timeout. After the timeout,
    it executes with whatever inputs have been received.

    Use cases:
    - Best-effort parallel execution
    - Handling potentially failing branches
    - Time-constrained workflows
    - Degraded operation when some services are slow

    The timeout can be:
    - Absolute: Start when merge is created, execute after N seconds
    - Relative: Start when first input arrives, execute after N seconds
    """

    def __init__(
        self, timeout_seconds: float, minimum_inputs: int = 1, timeout_mode: str = "relative"
    ) -> None:
        """
        Initialize TimeoutStrategy.

        Args:
            timeout_seconds: Timeout in seconds
            minimum_inputs: Minimum inputs required before allowing timeout
            timeout_mode: 'absolute' (from merge creation) or
                         'relative' (from first input)
        """
        super().__init__(name="timeout", description=f"Wait up to {timeout_seconds}s for inputs")
        self.timeout_seconds = timeout_seconds
        self.minimum_inputs = minimum_inputs
        self.timeout_mode = timeout_mode

        self._start_time: datetime | None = None
        self._first_input_time: datetime | None = None

    def should_execute(self, received_inputs: int, total_inputs: int, **kwargs) -> bool:
        """
        Execute when all inputs received OR timeout expires.

        Returns True if:
        1. All inputs received, OR
        2. Minimum inputs received AND timeout expired
        """
        # Initialize start time on first check
        if self._start_time is None:
            self._start_time = datetime.now()

        # Record first input time
        if received_inputs > 0 and self._first_input_time is None:
            self._first_input_time = datetime.now()

        # If we have all inputs, execute immediately
        if received_inputs >= total_inputs:
            return True

        # Check if minimum inputs requirement is met
        if received_inputs < self.minimum_inputs:
            return False

        # Check timeout
        current_time = datetime.now()

        if self.timeout_mode == "absolute":
            # Timeout from merge creation
            elapsed = (current_time - self._start_time).total_seconds()
        else:  # relative
            # Timeout from first input
            if self._first_input_time is None:
                return False
            elapsed = (current_time - self._first_input_time).total_seconds()

        return elapsed >= self.timeout_seconds

    def get_merge_mode(self) -> str:
        """Use all inputs that have arrived."""
        return "partial"

    def _get_elapsed(self) -> float | None:
        """Calculate elapsed time based on timeout mode."""
        if self._start_time is None:
            return None
        current_time = datetime.now()
        if self.timeout_mode == "relative" and self._first_input_time:
            return (current_time - self._first_input_time).total_seconds()
        return (current_time - self._start_time).total_seconds()

    def has_timed_out(self) -> bool:
        """Check if the timeout has expired."""
        elapsed = self._get_elapsed()
        return elapsed is not None and elapsed >= self.timeout_seconds

    def get_elapsed_time(self) -> float | None:
        """Get elapsed time in seconds."""
        return self._get_elapsed()

    def get_remaining_time(self) -> float | None:
        """Get remaining time until timeout in seconds."""
        elapsed = self._get_elapsed()
        if elapsed is None:
            return self.timeout_seconds
        return max(0, self.timeout_seconds - elapsed)

    def reset(self) -> None:
        """Reset timeout state."""
        self._start_time = None
        self._first_input_time = None
