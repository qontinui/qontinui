"""Enhanced State Transition with multi-state activation support.

This module implements the hybrid approach combining Brobot's comprehensive
state management with Qontinui's superior algorithms.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class StaysVisible(Enum):
    """Visibility behavior after transition."""

    NONE = "NONE"  # Inherit from StateTransitions container
    TRUE = "TRUE"  # Source state remains visible
    FALSE = "FALSE"  # Source state becomes hidden


@dataclass
class TransitionContext:
    """Context information for transition execution."""

    current_states: set[int]  # Currently active state IDs
    target_states: set[int]  # States to activate
    exit_states: set[int]  # States to exit
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the context."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata from the context."""
        return self.metadata.get(key, default)


@dataclass
class TransitionResult:
    """Result of a transition execution."""

    successful: bool
    activated_states: set[int] = field(default_factory=set)
    deactivated_states: set[int] = field(default_factory=set)
    hidden_states: set[int] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.successful = False

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0


@dataclass
class StateTransition(ABC):
    """Enhanced base class for state transitions with multi-state support.

    This hybrid implementation combines:
    - Brobot's multi-state activation capability
    - Brobot's comprehensive state management
    - Qontinui's clean Python design

    Key features:
    - Transitions can activate multiple states simultaneously
    - All activated states participate in pathfinding
    - Incoming transitions execute for all activated states
    - Path costs influence route selection
    """

    # Identification
    id: int = field(default_factory=lambda: id(object()))  # Unique transition ID

    # Source states (where transition can originate from)
    from_states: set[int] = field(default_factory=set)

    # Multi-state activation (from Brobot)
    activate: set[int] = field(default_factory=set)
    exit: set[int] = field(default_factory=set)

    # Visibility control
    stays_visible: StaysVisible = StaysVisible.NONE

    # Path-finding integration
    path_cost: int = 1  # Higher cost discourages this transition

    # Success tracking for reliability
    times_successful: int = 0
    times_failed: int = 0

    # Optional name and description
    name: str = ""
    description: str = ""

    @abstractmethod
    def execute(self, context: TransitionContext) -> TransitionResult:
        """Execute this transition with full state management.

        This method should:
        1. Perform the transition action
        2. Handle state activation/deactivation
        3. Manage visibility changes
        4. Track success/failure

        Args:
            context: Transition execution context

        Returns:
            TransitionResult with execution details
        """
        pass

    @abstractmethod
    def can_execute(self, context: TransitionContext) -> bool:
        """Check if this transition can be executed.

        Args:
            context: Transition execution context

        Returns:
            True if transition can be executed
        """
        pass

    def get_success_rate(self) -> float:
        """Calculate the success rate of this transition.

        Returns:
            Success rate between 0.0 and 1.0
        """
        total = self.times_successful + self.times_failed
        if total == 0:
            return 1.0  # Assume success if never executed
        return self.times_successful / total

    def record_success(self) -> None:
        """Record a successful execution."""
        self.times_successful += 1

    def record_failure(self) -> None:
        """Record a failed execution."""
        self.times_failed += 1

    def get_total_cost(self) -> float:
        """Get the total cost for pathfinding.

        Combines path cost with reliability factor.

        Returns:
            Total cost value
        """
        base_cost = float(self.path_cost)
        reliability_penalty = (1.0 - self.get_success_rate()) * 10.0
        return base_cost + reliability_penalty

    def activates_all(self, states: set[int]) -> bool:
        """Check if this transition activates all specified states.

        Args:
            states: Set of state IDs to check

        Returns:
            True if all states are activated by this transition
        """
        return states.issubset(self.activate)

    @property
    def score(self) -> int:
        """Get path-finding score as a property.

        Returns:
            Score value (path_cost)
        """
        return self.path_cost

    def __str__(self) -> str:
        """String representation."""
        name = self.name or self.__class__.__name__
        return (
            f"{name}(activate={len(self.activate)} states, "
            f"exit={len(self.exit)} states, cost={self.path_cost})"
        )


@dataclass
class TaskSequenceStateTransition(StateTransition):
    """Data-driven transition using a sequence of tasks.

    This transition type executes a predefined sequence of actions
    to move between states. It's declarative and can be defined
    in configuration files or built programmatically.
    """

    task_sequence: Optional["TaskSequence"] = None
    workflow_ids: list[str] = field(
        default_factory=list
    )  # Workflows to execute for this transition

    def execute(self, context: TransitionContext) -> TransitionResult:
        """Execute the task sequence.

        Args:
            context: Transition execution context

        Returns:
            TransitionResult with execution details
        """
        import time

        start_time = time.time()

        result = TransitionResult(successful=True)

        try:
            # Check if we can execute
            if not self.can_execute(context):
                result.add_error("Cannot execute transition: preconditions not met")
                self.record_failure()
                return result

            # Execute the task sequence if present
            if self.task_sequence:
                # This will be implemented when we port TaskSequence
                # For now, we'll simulate execution
                pass

            # Update states
            result.activated_states = self.activate.copy()
            result.deactivated_states = self.exit.copy()

            # Handle visibility
            if self.stays_visible == StaysVisible.FALSE:
                result.hidden_states = context.current_states - self.exit

            # Record success
            self.record_success()
            result.execution_time = time.time() - start_time

        except (RuntimeError, ValueError, TypeError) as e:
            result.add_error(f"Task sequence execution failed: {str(e)}")
            self.record_failure()

        return result

    def can_execute(self, context: TransitionContext) -> bool:
        """Check if the task sequence can be executed.

        Args:
            context: Transition execution context

        Returns:
            True if transition can be executed
        """
        # Check if we're in a valid starting state
        if not context.current_states:
            return False

        # Task sequence specific checks would go here
        return True


@dataclass
class CodeStateTransition(StateTransition):
    """Code-based transition using callable functions.

    This is the Python equivalent of Brobot's JavaStateTransition,
    using Python callables instead of Java's BooleanSupplier.
    """

    # Condition to check if transition is possible
    condition: Callable[[], bool] | None = None

    # Action to perform during transition
    action: Callable[[], None] | None = None

    # Optional context-aware versions
    condition_with_context: Callable[[TransitionContext], bool] | None = None
    action_with_context: Callable[[TransitionContext], None] | None = None

    def execute(self, context: TransitionContext) -> TransitionResult:
        """Execute the code-based transition.

        Args:
            context: Transition execution context

        Returns:
            TransitionResult with execution details
        """
        import time

        start_time = time.time()

        result = TransitionResult(successful=True)

        try:
            # Check if we can execute
            if not self.can_execute(context):
                result.add_error("Cannot execute transition: condition not met")
                self.record_failure()
                return result

            # Execute the action
            if self.action_with_context:
                self.action_with_context(context)
            elif self.action:
                self.action()

            # Update states
            result.activated_states = self.activate.copy()
            result.deactivated_states = self.exit.copy()

            # Handle visibility
            if self.stays_visible == StaysVisible.FALSE:
                result.hidden_states = context.current_states - self.exit

            # Record success
            self.record_success()
            result.execution_time = time.time() - start_time

        except (RuntimeError, ValueError, TypeError) as e:
            result.add_error(f"Code transition execution failed: {str(e)}")
            self.record_failure()

        return result

    def can_execute(self, context: TransitionContext) -> bool:
        """Check if the transition condition is met.

        Args:
            context: Transition execution context

        Returns:
            True if transition can be executed
        """
        try:
            # Check context-aware condition first
            if self.condition_with_context:
                return self.condition_with_context(context)

            # Check simple condition
            if self.condition:
                return self.condition()

            # No condition means always executable
            return True

        except (RuntimeError, ValueError, TypeError):
            # Condition check failed, transition cannot be executed
            return False


# Placeholder for TaskSequence - will be implemented later
class TaskSequence:
    """Placeholder for TaskSequence class."""

    pass
