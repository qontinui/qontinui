"""Composite hook for coordinating multiple hooks.

This module provides a facade for managing multiple execution hooks.
"""

import logging
from typing import Any

from ...config import Action
from .base import ExecutionHook

logger = logging.getLogger(__name__)


class CompositeHook(ExecutionHook):
    """Composite hook that delegates to multiple hooks.

    Allows combining multiple hooks into a single hook instance.
    Uses the Composite pattern to coordinate hook execution.

    Attributes:
        hooks: List of child hooks
    """

    def __init__(self, hooks: list[ExecutionHook] | None = None) -> None:
        """Initialize composite hook.

        Args:
            hooks: List of hooks to compose (optional)
        """
        self.hooks = hooks or []

    def add_hook(self, hook: ExecutionHook):
        """Add a hook to the composite.

        Args:
            hook: Hook to add
        """
        self.hooks.append(hook)

    def remove_hook(self, hook: ExecutionHook):
        """Remove a hook from the composite.

        Args:
            hook: Hook to remove
        """
        if hook in self.hooks:
            self.hooks.remove(hook)

    def before_action(self, action: Action, context: dict[str, Any]):
        """Call before_action on all child hooks."""
        for hook in self.hooks:
            try:
                hook.before_action(action, context)
            except Exception as e:
                logger.warning(f"Hook {type(hook).__name__}.before_action failed: {e}")

    def after_action(self, action: Action, context: dict[str, Any], result: dict[str, Any]):
        """Call after_action on all child hooks."""
        for hook in self.hooks:
            try:
                hook.after_action(action, context, result)
            except Exception as e:
                logger.warning(f"Hook {type(hook).__name__}.after_action failed: {e}")

    def on_error(self, action: Action, context: dict[str, Any], error: Exception):
        """Call on_error on all child hooks."""
        for hook in self.hooks:
            try:
                hook.on_error(action, context, error)
            except Exception as e:
                logger.warning(f"Hook {type(hook).__name__}.on_error failed: {e}")
