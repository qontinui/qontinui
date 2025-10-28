"""Conditional chain execution mode.

Executes actions based on conditions.
"""

import time
from typing import TYPE_CHECKING

from .base_mode import BaseChainMode

if TYPE_CHECKING:
    from ..chain_action import ChainAction


class ConditionalMode(BaseChainMode):
    """Execute actions conditionally based on their conditions.

    This mode checks each action's condition before execution.
    Actions without conditions are always executed. Skipped actions
    don't count as failures.
    """

    def execute(self, actions: list["ChainAction"]) -> bool:
        """Execute actions based on their conditions.

        Args:
            actions: List of ChainAction objects to execute

        Returns:
            True if all executed actions succeeded, False if any executed action failed
        """
        self._pause_before()

        for i, chain_action in enumerate(actions):
            self.current_index = i

            # Check condition before execution
            if not chain_action.should_execute():
                continue

            start_time = time.time()
            success = chain_action.execute()
            duration = time.time() - start_time

            self._record_action(chain_action, success, duration)

            if not success and self.options.stop_on_error:
                self._pause_after()
                return False

        self._pause_after()
        return True
