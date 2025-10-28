"""ChainAction - individual action in an action chain.

Handles execution of a single action with conditions, callbacks, and retry logic.
"""

import time
from collections.abc import Callable
from typing import Any

from ...action_interface import ActionInterface


class ChainAction:
    """Individual action in a chain.

    Encapsulates an action along with its target, execution conditions,
    success/failure callbacks, and retry settings.
    """

    def __init__(
        self,
        action: ActionInterface,
        target: Any | None = None,
        condition: Callable[[], bool] | None = None,
        on_success: Callable[[], None] | None = None,
        on_failure: Callable[[], None] | None = None,
        max_retries: int = 0,
    ) -> None:
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
            except (TypeError, ValueError, AttributeError):
                # Condition evaluation failed, skip this action
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
