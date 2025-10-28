"""Continue-on-error chain execution mode.

Executes all actions regardless of failures.
"""

import time
from typing import TYPE_CHECKING

from .base_mode import BaseChainMode

if TYPE_CHECKING:
    from ..chain_action import ChainAction


class ContinueMode(BaseChainMode):
    """Execute all actions, continuing even if errors occur.

    This mode executes every action in the chain regardless of
    whether previous actions succeeded or failed. Useful for
    cleanup operations or when partial success is acceptable.
    """

    def execute(self, actions: list["ChainAction"]) -> bool:
        """Execute all actions regardless of individual failures.

        Args:
            actions: List of ChainAction objects to execute

        Returns:
            True if at least one action succeeded, False if all failed
        """
        self._pause_before()

        any_success = False

        for i, chain_action in enumerate(actions):
            self.current_index = i

            start_time = time.time()
            success = chain_action.execute()
            duration = time.time() - start_time

            if success:
                any_success = True

            self._record_action(chain_action, success, duration)

        self._pause_after()
        return any_success
