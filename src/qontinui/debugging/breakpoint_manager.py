"""Breakpoint management for the debugging system.

This module provides the BreakpointManager class which handles
creation, removal, and evaluation of breakpoints.
"""

import threading
import uuid
from collections.abc import Callable
from typing import Any

from .types import Breakpoint, BreakpointType


class BreakpointManager:
    """Manages breakpoints for debugging sessions.

    The BreakpointManager maintains a collection of breakpoints and
    provides methods to add, remove, enable/disable, and check breakpoints.
    Thread-safe for concurrent access.
    """

    def __init__(self) -> None:
        """Initialize the breakpoint manager."""
        self._breakpoints: dict[str, Breakpoint] = {}
        self._lock = threading.RLock()

    def add_breakpoint(
        self,
        breakpoint_type: BreakpointType,
        action_id: str | None = None,
        action_type: str | None = None,
        condition: Callable[[Any], bool] | None = None,
        condition_str: str | None = None,
    ) -> str:
        """Add a new breakpoint.

        Args:
            breakpoint_type: Type of breakpoint to add
            action_id: Action ID to break on (for ACTION_ID type)
            action_type: Action type to break on (for ACTION_TYPE type)
            condition: Callable condition for conditional breakpoints
            condition_str: String representation of condition

        Returns:
            Breakpoint ID

        Raises:
            ValueError: If required parameters are missing for breakpoint type
        """
        # Validate parameters based on type
        if breakpoint_type == BreakpointType.ACTION_ID and not action_id:
            raise ValueError("action_id is required for ACTION_ID breakpoint")
        if breakpoint_type == BreakpointType.ACTION_TYPE and not action_type:
            raise ValueError("action_type is required for ACTION_TYPE breakpoint")
        if breakpoint_type in (BreakpointType.CONDITIONAL, BreakpointType.MATCH_COUNT):
            if not condition:
                raise ValueError(
                    f"condition is required for {breakpoint_type.value} breakpoint"
                )

        bp_id = str(uuid.uuid4())

        with self._lock:
            breakpoint = Breakpoint(
                id=bp_id,
                type=breakpoint_type,
                action_id=action_id,
                action_type=action_type,
                condition=condition,
                condition_str=condition_str,
            )
            self._breakpoints[bp_id] = breakpoint

        return bp_id

    def add_action_breakpoint(self, action_id: str) -> str:
        """Add a breakpoint for a specific action ID.

        Args:
            action_id: Action ID to break on

        Returns:
            Breakpoint ID
        """
        return self.add_breakpoint(BreakpointType.ACTION_ID, action_id=action_id)

    def add_type_breakpoint(self, action_type: str) -> str:
        """Add a breakpoint for a specific action type.

        Args:
            action_type: Action type to break on (e.g., "Click", "Find")

        Returns:
            Breakpoint ID
        """
        return self.add_breakpoint(BreakpointType.ACTION_TYPE, action_type=action_type)

    def add_conditional_breakpoint(
        self, condition: Callable[[Any], bool], condition_str: str | None = None
    ) -> str:
        """Add a conditional breakpoint.

        Args:
            condition: Callable that returns True when breakpoint should trigger
            condition_str: String representation of condition for display

        Returns:
            Breakpoint ID
        """
        return self.add_breakpoint(
            BreakpointType.CONDITIONAL, condition=condition, condition_str=condition_str
        )

    def add_error_breakpoint(self) -> str:
        """Add a breakpoint that triggers on any error.

        Returns:
            Breakpoint ID
        """
        return self.add_breakpoint(BreakpointType.ERROR)

    def remove_breakpoint(self, breakpoint_id: str) -> bool:
        """Remove a breakpoint.

        Args:
            breakpoint_id: ID of breakpoint to remove

        Returns:
            True if breakpoint was removed, False if not found
        """
        with self._lock:
            return self._breakpoints.pop(breakpoint_id, None) is not None

    def enable_breakpoint(self, breakpoint_id: str) -> bool:
        """Enable a breakpoint.

        Args:
            breakpoint_id: ID of breakpoint to enable

        Returns:
            True if breakpoint was found and enabled
        """
        with self._lock:
            bp = self._breakpoints.get(breakpoint_id)
            if bp:
                bp.enabled = True
                return True
            return False

    def disable_breakpoint(self, breakpoint_id: str) -> bool:
        """Disable a breakpoint.

        Args:
            breakpoint_id: ID of breakpoint to disable

        Returns:
            True if breakpoint was found and disabled
        """
        with self._lock:
            bp = self._breakpoints.get(breakpoint_id)
            if bp:
                bp.enabled = False
                return True
            return False

    def get_breakpoint(self, breakpoint_id: str) -> Breakpoint | None:
        """Get a breakpoint by ID.

        Args:
            breakpoint_id: Breakpoint ID

        Returns:
            Breakpoint if found, None otherwise
        """
        with self._lock:
            return self._breakpoints.get(breakpoint_id)

    def list_breakpoints(self) -> list[Breakpoint]:
        """List all breakpoints.

        Returns:
            List of all breakpoints
        """
        with self._lock:
            return list(self._breakpoints.values())

    def clear_all(self) -> int:
        """Remove all breakpoints.

        Returns:
            Number of breakpoints cleared
        """
        with self._lock:
            count = len(self._breakpoints)
            self._breakpoints.clear()
            return count

    def check_breakpoint(
        self, context: dict[str, Any]
    ) -> tuple[bool, list[Breakpoint]]:
        """Check if any breakpoint should trigger for the given context.

        Args:
            context: Execution context to check against breakpoints

        Returns:
            Tuple of (should_break, list of triggered breakpoints)
        """
        triggered = []

        with self._lock:
            for bp in self._breakpoints.values():
                if bp.should_break(context):
                    bp.hit_count += 1
                    triggered.append(bp)

        return (len(triggered) > 0, triggered)

    def get_statistics(self) -> dict[str, Any]:
        """Get breakpoint statistics.

        Returns:
            Dictionary containing statistics
        """
        with self._lock:
            total = len(self._breakpoints)
            enabled = sum(1 for bp in self._breakpoints.values() if bp.enabled)
            by_type = {}

            for bp in self._breakpoints.values():
                type_name = bp.type.value
                if type_name not in by_type:
                    by_type[type_name] = {"count": 0, "hits": 0}
                by_type[type_name]["count"] += 1
                by_type[type_name]["hits"] += bp.hit_count

            return {
                "total_breakpoints": total,
                "enabled_breakpoints": enabled,
                "disabled_breakpoints": total - enabled,
                "by_type": by_type,
            }

    def format_breakpoint(self, breakpoint: Breakpoint) -> str:
        """Format a breakpoint for display.

        Args:
            breakpoint: Breakpoint to format

        Returns:
            Formatted string representation
        """
        status = "enabled" if breakpoint.enabled else "disabled"
        type_info = ""

        if breakpoint.type == BreakpointType.ACTION_ID:
            type_info = f"action_id={breakpoint.action_id}"
        elif breakpoint.type == BreakpointType.ACTION_TYPE:
            type_info = f"action_type={breakpoint.action_type}"
        elif breakpoint.type == BreakpointType.CONDITIONAL:
            type_info = f"condition={breakpoint.condition_str or '<lambda>'}"
        elif breakpoint.type == BreakpointType.ERROR:
            type_info = "on_error"
        elif breakpoint.type == BreakpointType.MATCH_COUNT:
            type_info = f"match_count={breakpoint.condition_str or '<condition>'}"
        elif breakpoint.type == BreakpointType.STATE_CHANGE:
            type_info = "state_change"

        return (
            f"[{breakpoint.id[:8]}] {breakpoint.type.value} "
            f"({type_info}) [{status}] hits={breakpoint.hit_count}"
        )

    def __repr__(self) -> str:
        """String representation of breakpoint manager."""
        with self._lock:
            return f"BreakpointManager(breakpoints={len(self._breakpoints)})"
