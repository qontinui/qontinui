"""Base execution strategy for multiple actions."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..executor import TaskExecutor
    from ..task import ActionTask


class BaseExecutionStrategy(ABC):
    """Abstract base class for execution strategies."""

    def __init__(self, fail_fast: bool = False, record_actions: bool = True) -> None:
        """Initialize execution strategy.

        Args:
            fail_fast: Stop execution on first failure
            record_actions: Whether to record action execution
        """
        self.fail_fast = fail_fast
        self.record_actions = record_actions

    @abstractmethod
    def execute(
        self,
        tasks: list["ActionTask"],
        executor: "TaskExecutor",
    ) -> bool:
        """Execute tasks using this strategy.

        Args:
            tasks: List of tasks to execute
            executor: Task executor to use

        Returns:
            True if execution successful according to strategy rules
        """
        pass
