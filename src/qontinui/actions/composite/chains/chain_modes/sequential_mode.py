"""Sequential chain execution mode.

Executes actions in order, stopping on first failure.
"""

import time
from typing import TYPE_CHECKING

from .base_mode import BaseChainMode

if TYPE_CHECKING:
    from ..chain_action import ChainAction


class SequentialMode(BaseChainMode):
    """Execute actions sequentially, stop on first failure.

    This is the default execution mode. Actions execute in order,
    and the chain stops immediately if any action fails.
    """

    def execute(self, actions: list["ChainAction"]) -> bool:
        """Execute actions sequentially until completion or failure.

        Args:
            actions: List of ChainAction objects to execute

        Returns:
            True if all actions succeeded, False if any failed
        """
        self._pause_before()

        for i, chain_action in enumerate(actions):
            self.current_index = i

            start_time = time.time()
            success = chain_action.execute()
            duration = time.time() - start_time

            self._record_action(chain_action, success, duration)

            if not success and self.options.stop_on_error:
                self._pause_after()
                return False

        self._pause_after()
        return True
