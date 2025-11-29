"""ActionChain - ported from Qontinui framework.

Chains multiple actions together for sequential execution.
"""

from collections.abc import Callable
from enum import Enum, auto
from typing import Any

from ....model.action.action_record import ActionRecord
from ...action_config import ActionConfig
from ...action_interface import ActionInterface
from ...action_result import ActionResult
from ...action_type import ActionType
from ...object_collection import ObjectCollection
from .action_builders import ClickBuilder, DragBuilder, TypeBuilder
from .chain_action import ChainAction
from .chain_modes import (
    BaseChainMode,
    ConditionalMode,
    ContinueMode,
    RetryMode,
    SequentialMode,
)


class ChainMode(Enum):
    """Execution mode for action chains."""

    SEQUENTIAL = auto()  # Execute in order, stop on failure
    CONTINUE_ON_ERROR = auto()  # Continue even if action fails
    RETRY_ON_ERROR = auto()  # Retry failed actions
    CONDITIONAL = auto()  # Execute based on conditions


class ActionChainOptions(ActionConfig):
    """Options for action chains.

    Port of ActionChainOptions from Qontinui framework.
    """

    def __init__(self) -> None:
        """Initialize action chain options."""
        super().__init__()
        self.chain_mode: ChainMode = ChainMode.SEQUENTIAL
        self.stop_on_error: bool = True
        self.retry_failed: bool = False
        self.max_retries: int = 3
        self.retry_delay: float = 1.0
        self.parallel_execution: bool = False
        self.record_actions: bool = True

    def sequential(self) -> "ActionChainOptions":
        """Use sequential execution.

        Returns:
            Self for fluent interface
        """
        self.chain_mode = ChainMode.SEQUENTIAL
        self.stop_on_error = True
        return self

    def continue_on_error(self) -> "ActionChainOptions":
        """Continue execution on error.

        Returns:
            Self for fluent interface
        """
        self.chain_mode = ChainMode.CONTINUE_ON_ERROR
        self.stop_on_error = False
        return self

    def with_retries(self, max_retries: int) -> "ActionChainOptions":
        """Set maximum retries for failed actions.

        Args:
            max_retries: Maximum retry attempts

        Returns:
            Self for fluent interface
        """
        self.chain_mode = ChainMode.RETRY_ON_ERROR
        self.retry_failed = True
        self.max_retries = max_retries
        return self

    def conditional(self) -> "ActionChainOptions":
        """Use conditional execution.

        Returns:
            Self for fluent interface
        """
        self.chain_mode = ChainMode.CONDITIONAL
        return self


class ActionChain(ActionInterface):
    """Action chain implementation.

    Port of ActionChain from Qontinui framework class.

    Provides sequential or conditional execution of multiple
    actions with retry logic, error handling, and callbacks.
    """

    def __init__(self, options: ActionChainOptions | None = None) -> None:
        """Initialize ActionChain.

        Args:
            options: Chain options
        """
        self.options = options or ActionChainOptions()
        self._actions: list[ChainAction] = []
        self._mode: BaseChainMode = self._create_mode()

    def _create_mode(self) -> BaseChainMode:
        """Create the appropriate execution mode based on options.

        Returns:
            Execution mode instance
        """
        if self.options.chain_mode == ChainMode.SEQUENTIAL:
            return SequentialMode(self.options)
        elif self.options.chain_mode == ChainMode.CONTINUE_ON_ERROR:
            return ContinueMode(self.options)
        elif self.options.chain_mode == ChainMode.RETRY_ON_ERROR:
            return RetryMode(self.options)
        elif self.options.chain_mode == ChainMode.CONDITIONAL:
            return ConditionalMode(self.options)
        else:
            return SequentialMode(self.options)

    def get_action_type(self) -> ActionType:
        """Return the action type.

        Returns:
            ActionType for the first action in the chain, or FIND if chain is empty
        """
        if self._actions:
            first_action = self._actions[0].action
            if hasattr(first_action, "get_action_type"):
                return first_action.get_action_type()
        # Default to FIND as a generic action type
        return ActionType.FIND

    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Execute the action chain using the Qontinui framework pattern.

        Args:
            matches: Contains ActionOptions and accumulates execution results
            object_collections: Collections containing targets for the chain
        """
        # Execute the chain
        success = self.execute()

        # Update matches with results
        object.__setattr__(matches, "success", success)

        # Add execution history to matches
        for record in self._mode.execution_history:
            matches.add_execution_record(record)  # type: ignore[arg-type]

    def add(self, action: ActionInterface, target: Any | None = None, **kwargs) -> "ActionChain":
        """Add action to chain.

        Args:
            action: Action to add
            target: Target for action
            **kwargs: Additional ChainAction parameters

        Returns:
            Self for fluent interface
        """
        chain_action = ChainAction(action, target, **kwargs)
        self._actions.append(chain_action)
        return self

    def add_click(self, target: Any) -> "ActionChain":
        """Add click action to chain.

        Args:
            target: Click target

        Returns:
            Self for fluent interface
        """
        return self.add(ClickBuilder(), target)  # type: ignore[arg-type]

    def add_type(self, text: str, target: Any | None = None) -> "ActionChain":
        """Add type action to chain.

        Args:
            text: Text to type
            target: Optional target field

        Returns:
            Self for fluent interface
        """
        return self.add(TypeBuilder(text), target)  # type: ignore[arg-type]

    def add_wait(self, seconds: float) -> "ActionChain":
        """Add wait action to chain.

        Args:
            seconds: Seconds to wait

        Returns:
            Self for fluent interface
        """
        from ...basic.wait.wait import Wait, WaitOptions

        return self.add(Wait(WaitOptions().for_time(seconds)))

    def add_drag(self, start: Any, end: Any) -> "ActionChain":
        """Add drag action to chain.

        Args:
            start: Drag start
            end: Drag end

        Returns:
            Self for fluent interface
        """
        return self.add(DragBuilder(start, end))  # type: ignore[arg-type]

    def add_conditional(
        self,
        action: ActionInterface,
        condition: Callable[[], bool],
        target: Any | None = None,
    ) -> "ActionChain":
        """Add conditional action to chain.

        Args:
            action: Action to execute
            condition: Condition function
            target: Optional target

        Returns:
            Self for fluent interface
        """
        return self.add(action, target, condition=condition)

    def add_with_callbacks(
        self,
        action: ActionInterface,
        on_success: Callable[[], None],
        on_failure: Callable[[], None],
        target: Any | None = None,
    ) -> "ActionChain":
        """Add action with callbacks.

        Args:
            action: Action to execute
            on_success: Success callback
            on_failure: Failure callback
            target: Optional target

        Returns:
            Self for fluent interface
        """
        return self.add(action, target, on_success=on_success, on_failure=on_failure)

    def execute(self) -> bool:
        """Execute the action chain.

        Returns:
            True if chain executed successfully
        """
        # Clear execution history before new execution
        self._mode.execution_history.clear()
        self._mode.current_index = 0

        # Execute using the appropriate mode strategy
        return self._mode.execute(self._actions)

    def get_execution_history(self) -> list[ActionRecord]:
        """Get execution history.

        Returns:
            List of action records
        """
        return self._mode.execution_history.copy()

    def get_current_index(self) -> int:
        """Get current action index.

        Returns:
            Current index in chain
        """
        return self._mode.current_index

    def clear(self) -> "ActionChain":
        """Clear all actions from chain.

        Returns:
            Self for fluent interface
        """
        self._actions.clear()
        self._mode.execution_history.clear()
        self._mode.current_index = 0
        return self

    def size(self) -> int:
        """Get number of actions in chain.

        Returns:
            Number of actions
        """
        return len(self._actions)

    @staticmethod
    def create() -> "ActionChain":
        """Create new action chain.

        Returns:
            New ActionChain instance
        """
        return ActionChain()
