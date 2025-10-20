"""
Merge strategies for handling multiple incoming connections.

This module provides different strategies for determining when a merge node
should execute based on which input paths have arrived.
"""

from abc import ABC, abstractmethod
from datetime import datetime


class MergeStrategy(ABC):
    """
    Abstract base class for merge strategies.

    A merge strategy determines when a merge node should execute based on
    which input paths have completed.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            description: Human-readable description
        """
        self.name = name
        self.description = description

    @abstractmethod
    def should_execute(self, received_inputs: int, total_inputs: int, **kwargs) -> bool:
        """
        Determine if the merge node should execute.

        Args:
            received_inputs: Number of inputs received so far
            total_inputs: Total number of expected inputs
            **kwargs: Additional strategy-specific parameters

        Returns:
            True if the merge should execute now
        """
        pass

    def get_merge_mode(self) -> str:
        """
        Get the merge mode for context merging.

        Returns:
            One of: 'all', 'any', 'first'
        """
        return "all"  # Default to merging all contexts

    def should_wait_for_more(self, received_inputs: int, total_inputs: int, **kwargs) -> bool:
        """
        Determine if we should wait for more inputs.

        Args:
            received_inputs: Number of inputs received
            total_inputs: Total number of expected inputs
            **kwargs: Additional parameters

        Returns:
            True if should wait for more inputs
        """
        return not self.should_execute(received_inputs, total_inputs, **kwargs)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"


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

    def __init__(self):
        super().__init__(name="wait_all", description="Wait for all incoming paths to complete")

    def should_execute(self, received_inputs: int, total_inputs: int, **kwargs) -> bool:
        """Execute only when all inputs received."""
        return received_inputs >= total_inputs

    def get_merge_mode(self) -> str:
        """Merge all input contexts."""
        return "all"


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

    def __init__(self, allow_multiple_executions: bool = False):
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

    def __init__(self):
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
    ):
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

    def has_timed_out(self) -> bool:
        """Check if the timeout has expired."""
        if self._start_time is None:
            return False

        current_time = datetime.now()

        if self.timeout_mode == "absolute":
            elapsed = (current_time - self._start_time).total_seconds()
        else:
            if self._first_input_time is None:
                return False
            elapsed = (current_time - self._first_input_time).total_seconds()

        return elapsed >= self.timeout_seconds

    def get_elapsed_time(self) -> float | None:
        """Get elapsed time in seconds."""
        if self._start_time is None:
            return None

        current_time = datetime.now()

        if self.timeout_mode == "relative" and self._first_input_time:
            return (current_time - self._first_input_time).total_seconds()
        else:
            return (current_time - self._start_time).total_seconds()

    def get_remaining_time(self) -> float | None:
        """Get remaining time until timeout in seconds."""
        elapsed = self.get_elapsed_time()
        if elapsed is None:
            return self.timeout_seconds
        return max(0, self.timeout_seconds - elapsed)

    def reset(self) -> None:
        """Reset timeout state."""
        self._start_time = None
        self._first_input_time = None


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

    def __init__(self, threshold: float | None = None):
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


class CustomStrategy(MergeStrategy):
    """
    Custom strategy with user-defined logic.

    Allows implementing custom merge logic by providing a callable
    that determines when to execute.
    """

    def __init__(self, name: str, should_execute_func, description: str = "Custom merge strategy"):
        """
        Initialize CustomStrategy.

        Args:
            name: Strategy name
            should_execute_func: Callable that takes (received_inputs, total_inputs, **kwargs)
                                and returns bool
            description: Strategy description
        """
        super().__init__(name=name, description=description)
        self.should_execute_func = should_execute_func

    def should_execute(self, received_inputs: int, total_inputs: int, **kwargs) -> bool:
        """Execute based on custom function."""
        return self.should_execute_func(received_inputs, total_inputs, **kwargs)


# Registry of available strategies
STRATEGY_REGISTRY: dict[str, type[MergeStrategy]] = {
    "wait_all": WaitAllStrategy,
    "wait_any": WaitAnyStrategy,
    "wait_first": WaitFirstStrategy,
    "timeout": TimeoutStrategy,
    "majority": MajorityStrategy,
    "custom": CustomStrategy,
}


def create_strategy(strategy_type: str, **kwargs) -> MergeStrategy:
    """
    Create a merge strategy instance.

    Args:
        strategy_type: Type of strategy to create
        **kwargs: Strategy-specific parameters

    Returns:
        MergeStrategy instance

    Raises:
        ValueError: If strategy type is unknown
    """
    strategy_class = STRATEGY_REGISTRY.get(strategy_type)
    if strategy_class is None:
        raise ValueError(
            f"Unknown strategy type: {strategy_type}. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )

    return strategy_class(**kwargs)


def get_available_strategies() -> list[str]:
    """Get list of available strategy types."""
    return list(STRATEGY_REGISTRY.keys())
