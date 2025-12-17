"""Logging hooks for execution monitoring.

This module provides hooks for logging execution progress.
"""

import logging
from typing import Any

from ...config import Action
from .base import ExecutionHook

logger = logging.getLogger(__name__)


class LoggingHook(ExecutionHook):
    """Logs execution progress at various log levels.

    Attributes:
        logger_name: Name of logger to use
        log_level: Default log level
        log_context: Whether to log full context
    """

    def __init__(
        self,
        logger_name: str = "qontinui.execution",
        log_level: int = logging.INFO,
        log_context: bool = False,
    ) -> None:
        """Initialize logging hook.

        Args:
            logger_name: Name of logger to use
            log_level: Default log level (DEBUG, INFO, WARNING, ERROR)
            log_context: Whether to log full context (can be verbose)
        """
        self.logger = logging.getLogger(logger_name)
        self.log_level = log_level
        self.log_context = log_context

    def before_action(self, action: Action, context: dict[str, Any]):
        """Log before action execution."""
        msg = f"Executing action '{action.id}' (type={action.type})"
        if action.name:
            msg += f" - {action.name}"

        self.logger.log(self.log_level, msg)

        if self.log_context:
            self.logger.debug(f"Context: {context}")

    def after_action(self, action: Action, context: dict[str, Any], result: dict[str, Any]):
        """Log after successful execution."""
        success = result.get("success", False)
        msg = f"Action '{action.id}' completed: {'SUCCESS' if success else 'FAILED'}"

        self.logger.log(self.log_level, msg)

        if not success and "error" in result:
            self.logger.warning(f"Action '{action.id}' error: {result['error']}")

    def on_error(self, action: Action, context: dict[str, Any], error: Exception):
        """Log execution error."""
        self.logger.error(f"Action '{action.id}' raised exception: {type(error).__name__}: {error}")
