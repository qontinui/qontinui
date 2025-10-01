"""ActionChain - ported from Qontinui framework.

Chains multiple actions together for sequential execution.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from ....model.action.action_record import ActionRecord
from ...action_config import ActionConfig
from ...action_interface import ActionInterface
from ...action_result import ActionResult
from ...action_type import ActionType
from ...object_collection import ObjectCollection


class ChainMode(Enum):
    """Execution mode for action chains."""

    SEQUENTIAL = auto()  # Execute in order, stop on failure
    CONTINUE_ON_ERROR = auto()  # Continue even if action fails
    RETRY_ON_ERROR = auto()  # Retry failed actions
    CONDITIONAL = auto()  # Execute based on conditions


class ChainAction:
    """Individual action in a chain."""

    def __init__(
        self,
        action: ActionInterface,
        target: Any | None = None,
        condition: Callable[[], bool] | None = None,
        on_success: Callable[[], None] | None = None,
        on_failure: Callable[[], None] | None = None,
        max_retries: int = 0,
    ):
        """Initialize chain action.

        Args:
            action: Action to execute
            target: Target for action
            condition: Condition to check before execution
            on_success: Callback on success
            on_failure: Callback on failure
            max_retries: Maximum retry attempts
        """
        self.action = action
        self.target = target
        self.condition = condition
        self.on_success = on_success
        self.on_failure = on_failure
        self.max_retries = max_retries
        self.execution_count = 0
        self.last_result = None

    def should_execute(self) -> bool:
        """Check if action should execute.

        Returns:
            True if should execute
        """
        if self.condition:
            try:
                return self.condition()
            except Exception:
                return False
        return True

    def execute(self) -> bool:
        """Execute the action.

        Returns:
            True if successful
        """
        if not self.should_execute():
            return True  # Skip but don't fail chain

        attempts = 0
        while attempts <= self.max_retries:
            try:
                self.execution_count += 1

                # Execute action with target
                if self.target is not None:
                    if hasattr(self.action, "execute"):
                        result = self.action.execute(self.target)
                    elif callable(self.action):
                        result = self.action(self.target)  # type: ignore
                    else:
                        raise RuntimeError(f"Action {self.action} is not executable")
                else:
                    if hasattr(self.action, "execute"):
                        result = self.action.execute()
                    elif callable(self.action):
                        result = self.action()  # type: ignore
                    else:
                        raise RuntimeError(f"Action {self.action} is not executable")

                self.last_result = result

                if result:
                    if self.on_success:
                        self.on_success()
                    return True

            except Exception as e:
                print(f"Action execution error: {e}")
                result = False

            attempts += 1
            if attempts <= self.max_retries:
                time.sleep(0.5)  # Brief pause before retry

        if self.on_failure:
            self.on_failure()
        return False


@dataclass
class ActionChainOptions(ActionConfig):
    """Options for action chains.

    Port of ActionChainOptions from Qontinui framework.
    """

    chain_mode: ChainMode = ChainMode.SEQUENTIAL
    stop_on_error: bool = True
    retry_failed: bool = False
    max_retries: int = 3
    retry_delay: float = 1.0
    parallel_execution: bool = False
    record_actions: bool = True

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

    def __init__(self, options: ActionChainOptions | None = None):
        """Initialize ActionChain.

        Args:
            options: Chain options
        """
        self.options = options or ActionChainOptions()
        self._actions: list[ChainAction] = []
        self._execution_history: list[ActionRecord] = []
        self._current_index = 0
        self._chain_result = True

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
        matches.success = success

        # Add execution history to matches
        for record in self._execution_history:
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
        from ...basic.click.click import Click

        return self.add(Click(), target)

    def add_type(self, text: str, target: Any | None = None) -> "ActionChain":
        """Add type action to chain.

        Args:
            text: Text to type
            target: Optional target field

        Returns:
            Self for fluent interface
        """
        from ...basic.type.type_action import TypeAction

        # Create a wrapper that passes text properly
        class TypeWrapper:
            def __init__(self, text: str):
                self.text = text
                self.action = TypeAction()

            def execute(self, target=None):
                return self.action.execute(self.text, target)

        return self.add(TypeWrapper(text), target)  # type: ignore[arg-type]

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
        from ..drag.drag import Drag

        # Create a wrapper for drag with both parameters
        class DragWrapper:
            def __init__(self, start: Any, end: Any):
                self.start = start
                self.end = end
                self.action = Drag()

            def execute(self):
                return self.action.execute(self.start, self.end)

        return self.add(DragWrapper(start, end))  # type: ignore[arg-type]

    def add_conditional(
        self, action: ActionInterface, condition: Callable[[], bool], target: Any | None = None
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
        self._current_index = 0
        self._chain_result = True
        self._execution_history.clear()

        # Apply pre-action pause
        self._pause_before()

        # Execute actions based on mode
        if self.options.chain_mode == ChainMode.SEQUENTIAL:
            self._chain_result = self._execute_sequential()
        elif self.options.chain_mode == ChainMode.CONTINUE_ON_ERROR:
            self._chain_result = self._execute_continue_on_error()
        elif self.options.chain_mode == ChainMode.RETRY_ON_ERROR:
            self._chain_result = self._execute_with_retries()
        elif self.options.chain_mode == ChainMode.CONDITIONAL:
            self._chain_result = self._execute_conditional()

        # Apply post-action pause
        self._pause_after()

        return self._chain_result

    def _execute_sequential(self) -> bool:
        """Execute actions sequentially.

        Returns:
            True if all succeeded
        """
        for i, chain_action in enumerate(self._actions):
            self._current_index = i

            start_time = time.time()
            success = chain_action.execute()
            duration = time.time() - start_time

            # Record action
            if self.options.record_actions:
                self._record_action(chain_action, success, duration)

            if not success and self.options.stop_on_error:
                return False

        return True

    def _execute_continue_on_error(self) -> bool:
        """Execute all actions regardless of errors.

        Returns:
            True if any succeeded
        """
        any_success = False

        for i, chain_action in enumerate(self._actions):
            self._current_index = i

            start_time = time.time()
            success = chain_action.execute()
            duration = time.time() - start_time

            if success:
                any_success = True

            # Record action
            if self.options.record_actions:
                self._record_action(chain_action, success, duration)

        return any_success

    def _execute_with_retries(self) -> bool:
        """Execute with retry logic.

        Returns:
            True if all eventually succeeded
        """
        for i, chain_action in enumerate(self._actions):
            self._current_index = i

            # Set max retries for this action
            chain_action.max_retries = self.options.max_retries

            start_time = time.time()
            success = chain_action.execute()
            duration = time.time() - start_time

            # Record action
            if self.options.record_actions:
                self._record_action(chain_action, success, duration)

            if not success:
                return False

        return True

    def _execute_conditional(self) -> bool:
        """Execute actions conditionally.

        Returns:
            True if executed successfully
        """
        for i, chain_action in enumerate(self._actions):
            self._current_index = i

            # Check condition
            if not chain_action.should_execute():
                continue

            start_time = time.time()
            success = chain_action.execute()
            duration = time.time() - start_time

            # Record action
            if self.options.record_actions:
                self._record_action(chain_action, success, duration)

            if not success and self.options.stop_on_error:
                return False

        return True

    def _record_action(self, chain_action: ChainAction, success: bool, duration: float):
        """Record action execution.

        Args:
            chain_action: Executed action
            success: Whether successful
            duration: Execution duration
        """
        record = ActionRecord(
            action_config=getattr(chain_action.action, "options", None),
            text=f"Chain action {self._current_index + 1}",
            duration=duration,
            action_success=success,
        )
        self._execution_history.append(record)

    def _pause_before(self):
        """Apply pre-action pause from options."""
        if self.options.pause_before > 0:
            time.sleep(self.options.pause_before)

    def _pause_after(self):
        """Apply post-action pause from options."""
        if self.options.pause_after > 0:
            time.sleep(self.options.pause_after)

    def get_execution_history(self) -> list[ActionRecord]:
        """Get execution history.

        Returns:
            List of action records
        """
        return self._execution_history.copy()

    def get_current_index(self) -> int:
        """Get current action index.

        Returns:
            Current index in chain
        """
        return self._current_index

    def clear(self) -> "ActionChain":
        """Clear all actions from chain.

        Returns:
            Self for fluent interface
        """
        self._actions.clear()
        self._execution_history.clear()
        self._current_index = 0
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
