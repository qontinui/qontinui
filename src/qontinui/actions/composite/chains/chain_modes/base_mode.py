"""Base chain execution mode.

Defines the abstract interface for chain execution strategies.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .....model.action.action_record import ActionRecord

if TYPE_CHECKING:
    from ..action_chain import ActionChainOptions
    from ..chain_action import ChainAction


class BaseChainMode(ABC):
    """Abstract base class for chain execution modes.

    Uses the Strategy pattern to allow different execution behaviors
    for action chains (sequential, retry, continue on error, etc.).
    """

    def __init__(self, options: ActionChainOptions) -> None:
        """Initialize chain mode.

        Args:
            options: Chain configuration options
        """
        self.options = options
        self.execution_history: list[ActionRecord] = []
        self.current_index = 0

    @abstractmethod
    def execute(self, actions: list[ChainAction]) -> bool:
        """Execute the chain of actions.

        Args:
            actions: List of ChainAction objects to execute

        Returns:
            True if chain execution was successful according to mode rules
        """
        pass

    def _record_action(
        self, chain_action: ChainAction, success: bool, duration: float
    ) -> None:
        """Record action execution in history.

        Args:
            chain_action: The executed action
            success: Whether execution was successful
            duration: Execution duration in seconds
        """
        if not self.options.record_actions:
            return

        record = ActionRecord(
            action_config=getattr(chain_action.action, "options", None),
            text=f"Chain action {self.current_index + 1}",
            duration=duration,
            action_success=success,
        )
        self.execution_history.append(record)

    def _pause_before(self) -> None:
        """Apply pre-action pause from options."""
        if self.options.pause_before_begin > 0:
            time.sleep(self.options.pause_before_begin)

    def _pause_after(self) -> None:
        """Apply post-action pause from options."""
        if self.options.pause_after_end > 0:
            time.sleep(self.options.pause_after_end)
