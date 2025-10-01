"""Wait action - ported from Qontinui framework.

Wait for conditions or time periods.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, cast

from ...action_config import ActionConfig
from ...action_interface import ActionInterface
from ...action_result import ActionResult
from ...action_type import ActionType
from ...object_collection import ObjectCollection


class WaitType(Enum):
    """Types of wait operations."""

    TIME = auto()  # Wait for specific time
    CONDITION = auto()  # Wait for condition
    VISIBLE = auto()  # Wait for element visible
    VANISH = auto()  # Wait for element to vanish
    CHANGE = auto()  # Wait for change in region
    STABLE = auto()  # Wait for stable state


@dataclass
class WaitOptions(ActionConfig):
    """Options for wait actions.

    Port of WaitOptions from Qontinui framework class.
    """

    wait_type: WaitType = WaitType.TIME
    timeout: float = 30.0  # Maximum wait time
    poll_interval: float = 0.5  # How often to check condition
    min_stability_time: float = 1.0  # For stable wait
    change_threshold: float = 0.05  # For change detection

    def for_time(self, seconds: float) -> "WaitOptions":
        """Wait for specific time.

        Args:
            seconds: Time to wait

        Returns:
            Self for fluent interface
        """
        self.wait_type = WaitType.TIME
        self.timeout = seconds
        return self

    def for_condition(self) -> "WaitOptions":
        """Wait for condition.

        Returns:
            Self for fluent interface
        """
        self.wait_type = WaitType.CONDITION
        return self

    def for_visible(self) -> "WaitOptions":
        """Wait for element to be visible.

        Returns:
            Self for fluent interface
        """
        self.wait_type = WaitType.VISIBLE
        return self

    def for_vanish(self) -> "WaitOptions":
        """Wait for element to vanish.

        Returns:
            Self for fluent interface
        """
        self.wait_type = WaitType.VANISH
        return self

    def for_change(self) -> "WaitOptions":
        """Wait for change in region.

        Returns:
            Self for fluent interface
        """
        self.wait_type = WaitType.CHANGE
        return self

    def for_stable(self) -> "WaitOptions":
        """Wait for stable state.

        Returns:
            Self for fluent interface
        """
        self.wait_type = WaitType.STABLE
        return self

    def with_timeout(self, seconds: float) -> "WaitOptions":
        """Set timeout.

        Args:
            seconds: Timeout in seconds

        Returns:
            Self for fluent interface
        """
        self.timeout = seconds
        return self

    def with_poll_interval(self, seconds: float) -> "WaitOptions":
        """Set poll interval.

        Args:
            seconds: Poll interval in seconds

        Returns:
            Self for fluent interface
        """
        self.poll_interval = seconds
        return self


class Wait(ActionInterface):
    """Wait action implementation.

    Port of Wait from Qontinui framework class.

    Provides various wait operations including time-based waits,
    condition waits, and element visibility waits.
    """

    def __init__(self, options: WaitOptions | None = None):
        """Initialize Wait action.

        Args:
            options: Wait options
        """
        self.options = options or WaitOptions()
        self._start_time: float | None = None
        self._elapsed_time: float = 0.0

    def get_action_type(self) -> ActionType:
        """Return the action type.

        Returns:
            ActionType.VANISH for vanish waits, otherwise a wait-related type
        """
        if self.options.wait_type == WaitType.VANISH:
            return ActionType.VANISH
        # For other wait types, we'll use VANISH as the closest match
        # TODO: Add more specific ActionType values for different wait types if needed
        return ActionType.VANISH

    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Execute the wait action using the Qontinui framework pattern.

        Args:
            matches: Contains ActionOptions and accumulates execution results
            object_collections: Collections containing targets to wait for
        """
        # Extract target from object collections if provided
        target = None
        if object_collections:
            collection = object_collections[0]
            if collection.matches:
                target = collection.matches[0]

        # Execute the wait action
        success = self.execute(target=target)

        # Update matches with results
        matches.success = success

    def execute(
        self, target: Any | None = None, condition: Callable[[], bool] | None = None
    ) -> bool:
        """Execute wait action.

        Args:
            target: Optional target element for visibility waits
            condition: Optional condition function for condition waits

        Returns:
            True if wait completed successfully
        """
        self._start_time = time.time()

        # Apply pre-action pause
        self._pause_before()

        result = False

        if self.options.wait_type == WaitType.TIME:
            result = self._wait_time()
        elif self.options.wait_type == WaitType.CONDITION:
            result = self._wait_condition(condition)
        elif self.options.wait_type == WaitType.VISIBLE:
            result = self._wait_visible(target)
        elif self.options.wait_type == WaitType.VANISH:
            result = self._wait_vanish(target)
        elif self.options.wait_type == WaitType.CHANGE:
            result = self._wait_change(target)
        elif self.options.wait_type == WaitType.STABLE:
            result = self._wait_stable(target)

        self._elapsed_time = time.time() - self._start_time

        # Apply post-action pause
        self._pause_after()

        return result

    def _wait_time(self) -> bool:
        """Wait for specified time.

        Returns:
            Always True
        """
        time.sleep(self.options.timeout)
        return True

    def _wait_condition(self, condition: Callable[[], bool] | None) -> bool:
        """Wait for condition to be true.

        Args:
            condition: Condition function

        Returns:
            True if condition met before timeout
        """
        if not condition:
            return False

        end_time = time.time() + self.options.timeout

        while time.time() < end_time:
            try:
                if condition():
                    return True
            except Exception as e:
                print(f"Condition check error: {e}")

            time.sleep(self.options.poll_interval)

        return False

    def _wait_visible(self, target: Any) -> bool:
        """Wait for element to be visible.

        Args:
            target: Target element

        Returns:
            True if visible before timeout
        """
        if not target:
            return False

        def check_visible():
            # This would use find to check visibility
            # For now, simulate with random success
            import random

            return random.random() > 0.7

        return self._wait_condition(check_visible)

    def _wait_vanish(self, target: Any) -> bool:
        """Wait for element to vanish.

        Args:
            target: Target element

        Returns:
            True if vanished before timeout
        """
        if not target:
            return False

        def check_vanished():
            # This would use find to check if element is gone
            # For now, simulate with random success
            import random

            return random.random() > 0.7

        return self._wait_condition(check_vanished)

    def _wait_change(self, target: Any) -> bool:
        """Wait for change in region.

        Args:
            target: Target region

        Returns:
            True if change detected before timeout
        """
        if not target:
            return False

        # Capture initial state
        initial_state = self._capture_state(target)

        def check_changed():
            current_state = self._capture_state(target)
            return self._has_changed(initial_state, current_state)

        return self._wait_condition(check_changed)

    def _wait_stable(self, target: Any) -> bool:
        """Wait for stable state (no changes).

        Args:
            target: Target region

        Returns:
            True if stable for required time
        """
        if not target:
            return False

        stable_start: float = time.time()
        last_state = None

        end_time = time.time() + self.options.timeout

        while time.time() < end_time:
            current_state = self._capture_state(target)

            if last_state is None:
                last_state = current_state
                stable_start = time.time()
            elif self._has_changed(last_state, current_state):
                # Reset stability timer
                last_state = current_state
                stable_start = time.time()
            elif time.time() - stable_start >= self.options.min_stability_time:
                # Stable for required time
                return True

            time.sleep(self.options.poll_interval)

        return False

    def _capture_state(self, target: Any) -> Any:
        """Capture current state of target.

        Args:
            target: Target to capture

        Returns:
            State representation
        """
        # This would capture screenshot or state
        # For now, return timestamp as mock state
        return time.time()

    def _has_changed(self, state1: Any, state2: Any) -> bool:
        """Check if state has changed.

        Args:
            state1: First state
            state2: Second state

        Returns:
            True if changed
        """
        # This would compare states/images
        # For now, use simple comparison
        return cast(bool, abs(state1 - state2) > self.options.change_threshold)

    def _pause_before(self):
        """Apply pre-action pause from options."""
        if self.options.pause_before > 0:
            time.sleep(self.options.pause_before)

    def _pause_after(self):
        """Apply post-action pause from options."""
        if self.options.pause_after > 0:
            time.sleep(self.options.pause_after)

    def get_elapsed_time(self) -> float:
        """Get elapsed wait time.

        Returns:
            Elapsed time in seconds
        """
        return self._elapsed_time

    @staticmethod
    def wait_seconds(seconds: float) -> bool:
        """Convenience method for time wait.

        Args:
            seconds: Seconds to wait

        Returns:
            Always True
        """
        return Wait(WaitOptions().for_time(seconds)).execute()

    @staticmethod
    def wait_until(condition: Callable[[], bool], timeout: float = 30.0) -> bool:
        """Convenience method for condition wait.

        Args:
            condition: Condition function
            timeout: Timeout in seconds

        Returns:
            True if condition met
        """
        options = WaitOptions().for_condition().with_timeout(timeout)
        return Wait(options).execute(condition=condition)
