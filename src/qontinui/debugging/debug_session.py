"""Debug session for managing execution state and debugging operations.

This module provides the DebugSession class which manages the state
and execution flow of a single debugging session.
"""

import threading
import uuid
from datetime import datetime
from typing import Any

from .types import ExecutionState, StepMode, VariableSnapshot


class DebugSession:
    """Manages a single debugging session with execution state and variables.

    A debug session represents a single debugging context with its own
    execution state, variable snapshots, and step control. Sessions are
    thread-safe to support concurrent debugging operations.
    """

    def __init__(self, name: str = "") -> None:
        """Initialize a debug session.

        Args:
            name: Human-readable name for this session
        """
        self.id = str(uuid.uuid4())
        self.name = name or f"Session-{self.id[:8]}"
        self.created_at = datetime.now()

        self._state = ExecutionState.IDLE
        self._step_mode: StepMode | None = None
        self._action_depth = 0

        # Variable snapshots indexed by action_id
        self._snapshots: dict[str, VariableSnapshot] = {}

        # Current action tracking
        self._current_action_id: str | None = None
        self._action_stack: list[str] = []

        # Step control
        self._should_pause = False
        self._step_target_depth: int | None = None

        # Thread safety
        self._lock = threading.RLock()

        # Wait condition for step control
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start unpaused

    @property
    def state(self) -> ExecutionState:
        """Get current execution state."""
        with self._lock:
            return self._state

    @state.setter
    def state(self, value: ExecutionState) -> None:
        """Set execution state."""
        with self._lock:
            self._state = value

    @property
    def current_action_id(self) -> str | None:
        """Get current action ID."""
        with self._lock:
            return self._current_action_id

    @property
    def action_depth(self) -> int:
        """Get current action nesting depth."""
        with self._lock:
            return self._action_depth

    def start_action(self, action_id: str) -> None:
        """Mark the start of an action.

        Args:
            action_id: Unique identifier for the action
        """
        with self._lock:
            self._current_action_id = action_id
            self._action_stack.append(action_id)
            self._action_depth = len(self._action_stack)

    def end_action(self, action_id: str) -> None:
        """Mark the end of an action.

        Args:
            action_id: Unique identifier for the action
        """
        with self._lock:
            if self._action_stack and self._action_stack[-1] == action_id:
                self._action_stack.pop()
                self._action_depth = len(self._action_stack)
                self._current_action_id = (
                    self._action_stack[-1] if self._action_stack else None
                )

    def snapshot_variables(self, action_id: str, variables: dict[str, Any]) -> None:
        """Store a snapshot of variables at the current execution point.

        Args:
            action_id: Action ID for this snapshot
            variables: Dictionary of variable names and values
        """
        with self._lock:
            snapshot = VariableSnapshot(
                timestamp=datetime.now(),
                action_id=action_id,
                variables=variables.copy(),
            )
            self._snapshots[action_id] = snapshot

    def get_snapshot(self, action_id: str) -> VariableSnapshot | None:
        """Retrieve a variable snapshot.

        Args:
            action_id: Action ID to retrieve snapshot for

        Returns:
            VariableSnapshot if found, None otherwise
        """
        with self._lock:
            return self._snapshots.get(action_id)

    def get_all_snapshots(self) -> list[VariableSnapshot]:
        """Get all variable snapshots for this session.

        Returns:
            List of all snapshots ordered by timestamp
        """
        with self._lock:
            return sorted(self._snapshots.values(), key=lambda s: s.timestamp)

    def step(self, mode: StepMode = StepMode.OVER) -> None:
        """Execute a single step in the specified mode.

        Args:
            mode: Step mode (into, over, out)
        """
        with self._lock:
            self._step_mode = mode
            self._state = ExecutionState.STEPPING

            # Set target depth based on mode
            if mode == StepMode.INTO:
                # Allow any depth
                self._step_target_depth = None
            elif mode == StepMode.OVER:
                # Stay at current depth
                self._step_target_depth = self._action_depth
            elif mode == StepMode.OUT:
                # Go to parent depth
                self._step_target_depth = max(0, self._action_depth - 1)

            # Allow one step
            self._pause_event.set()

    def should_pause_at_depth(self, depth: int) -> bool:
        """Check if execution should pause at the given depth.

        Args:
            depth: Current action depth

        Returns:
            True if should pause
        """
        with self._lock:
            if self._state != ExecutionState.STEPPING:
                return False

            if self._step_mode == StepMode.INTO:
                # Pause at next action regardless of depth
                return True
            elif self._step_mode == StepMode.OVER:
                # Pause when we return to target depth
                return depth <= (self._step_target_depth or 0)
            elif self._step_mode == StepMode.OUT:
                # Pause when we're above target depth
                return depth <= (self._step_target_depth or 0)

            return False

    def pause(self) -> None:
        """Pause execution."""
        with self._lock:
            self._should_pause = True
            self._state = ExecutionState.PAUSED
            self._pause_event.clear()

    def continue_execution(self) -> None:
        """Continue execution from paused state."""
        with self._lock:
            self._should_pause = False
            self._step_mode = None
            self._state = ExecutionState.RUNNING
            self._pause_event.set()

    def wait_if_paused(self, timeout: float | None = None) -> bool:
        """Wait if session is paused.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if event was set, False if timeout occurred
        """
        return self._pause_event.wait(timeout)

    def check_pause(self) -> None:
        """Check if execution should pause and wait if necessary.

        This should be called at the beginning of each action.
        """
        with self._lock:
            should_stop = False

            # Check if we should pause for stepping
            if self._state == ExecutionState.STEPPING:
                if self.should_pause_at_depth(self._action_depth):
                    self._state = ExecutionState.PAUSED
                    self._pause_event.clear()
                    should_stop = True

            # Check manual pause
            if self._should_pause:
                self._state = ExecutionState.PAUSED
                self._pause_event.clear()
                should_stop = True

        # Wait outside the lock to avoid deadlock
        if should_stop:
            self._pause_event.wait()

    def complete(self) -> None:
        """Mark the session as completed."""
        with self._lock:
            self._state = ExecutionState.COMPLETED
            self._pause_event.set()

    def error(self) -> None:
        """Mark the session as having an error."""
        with self._lock:
            self._state = ExecutionState.ERROR
            self._pause_event.set()

    def get_info(self) -> dict[str, Any]:
        """Get session information.

        Returns:
            Dictionary containing session details
        """
        with self._lock:
            return {
                "id": self.id,
                "name": self.name,
                "state": self._state.value,
                "created_at": self.created_at.isoformat(),
                "current_action": self._current_action_id,
                "action_depth": self._action_depth,
                "snapshot_count": len(self._snapshots),
            }

    def __repr__(self) -> str:
        """String representation of session."""
        return f"DebugSession(id={self.id[:8]}, name={self.name}, state={self._state.value})"
