"""Monitoring hooks for performance and progress tracking.

This module provides hooks for tracking execution progress and timing.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

from ...config import Action
from .base import ExecutionHook

logger = logging.getLogger(__name__)


class ProgressHook(ExecutionHook):
    """Reports execution progress as percentage.

    Attributes:
        total_actions: Total number of actions in workflow
        completed_actions: Number of actions completed
        progress_callback: Optional callback function for progress updates
    """

    def __init__(self, total_actions: int, progress_callback: Callable | None = None) -> None:
        """Initialize progress hook.

        Args:
            total_actions: Total number of actions in workflow
            progress_callback: Optional callback(action_id, progress_percent)
        """
        self.total_actions = total_actions
        self.completed_actions = 0
        self.progress_callback = progress_callback

    def before_action(self, action: Action, context: dict[str, Any]):
        """Track action start."""
        pass

    def after_action(self, action: Action, context: dict[str, Any], result: dict[str, Any]):
        """Update progress after action completion."""
        self.completed_actions += 1
        progress_percent = (self.completed_actions / self.total_actions) * 100

        logger.info(
            f"Progress: {progress_percent:.1f}% ({self.completed_actions}/{self.total_actions})"
        )

        if self.progress_callback:
            try:
                self.progress_callback(action.id, progress_percent)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def on_error(self, action: Action, context: dict[str, Any], error: Exception):
        """Count failed actions as completed for progress."""
        self.completed_actions += 1

    def get_progress(self) -> float:
        """Get current progress percentage.

        Returns:
            Progress as percentage (0-100)
        """
        if self.total_actions == 0:
            return 100.0
        return (self.completed_actions / self.total_actions) * 100


class TimingHook(ExecutionHook):
    """Tracks execution timing for performance analysis.

    Attributes:
        action_timings: Dictionary mapping action_id to execution time
        start_times: Dictionary tracking current action start times
    """

    def __init__(self) -> None:
        """Initialize timing hook."""
        self.action_timings: dict[str, float] = {}
        self.start_times: dict[str, float] = {}

    def before_action(self, action: Action, context: dict[str, Any]):
        """Record action start time."""
        self.start_times[action.id] = time.time()

    def after_action(self, action: Action, context: dict[str, Any], result: dict[str, Any]):
        """Record action completion time."""
        if action.id in self.start_times:
            elapsed = time.time() - self.start_times[action.id]
            self.action_timings[action.id] = elapsed
            logger.debug(f"Action '{action.id}' executed in {elapsed:.3f}s")
            del self.start_times[action.id]

    def on_error(self, action: Action, context: dict[str, Any], error: Exception):
        """Record timing even on error."""
        if action.id in self.start_times:
            elapsed = time.time() - self.start_times[action.id]
            self.action_timings[action.id] = elapsed
            del self.start_times[action.id]

    def get_timing_report(self) -> dict[str, Any]:
        """Get timing analysis report.

        Returns:
            Dictionary with timing statistics
        """
        if not self.action_timings:
            return {"error": "No timing data available"}

        total_time = sum(self.action_timings.values())
        avg_time = total_time / len(self.action_timings)

        sorted_timings = sorted(self.action_timings.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_actions": len(self.action_timings),
            "total_time": total_time,
            "average_time": avg_time,
            "slowest_actions": sorted_timings[:10],
            "fastest_actions": sorted_timings[-10:],
            "all_timings": self.action_timings,
        }
