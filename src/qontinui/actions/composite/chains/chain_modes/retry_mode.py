"""Retry chain execution mode.

Executes actions with automatic retry on failure.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from .base_mode import BaseChainMode

if TYPE_CHECKING:
    from ..chain_action import ChainAction


class RetryMode(BaseChainMode):
    """Execute actions with retry logic on failure.

    This mode automatically retries failed actions up to max_retries times.
    The entire chain must succeed (all actions eventually succeed) or the
    chain fails.
    """

    def execute(self, actions: list[ChainAction]) -> bool:
        """Execute actions with automatic retry on failure.

        Args:
            actions: List of ChainAction objects to execute

        Returns:
            True if all actions eventually succeeded, False if any failed after retries
        """
        self._pause_before()

        for i, chain_action in enumerate(actions):
            self.current_index = i

            # Set max retries for this action
            chain_action.max_retries = self.options.max_retries

            start_time = time.time()
            success = chain_action.execute()
            duration = time.time() - start_time

            self._record_action(chain_action, success, duration)

            if not success:
                self._pause_after()
                return False

        self._pause_after()
        return True
