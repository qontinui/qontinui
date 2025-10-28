"""Basic implementation of ExecutionController protocol.

Provides a thread-safe implementation of the ExecutionController interface
for use in pause/resume/stop operations.
"""

import asyncio
import logging
import threading

from .execution_controller import ExecutionController, ExecutionStoppedException
from .execution_state import ExecutionState

logger = logging.getLogger(__name__)


class BasicExecutionController(ExecutionController):
    """Basic thread-safe implementation of ExecutionController.

    This implementation provides:
    - Thread-safe state transitions
    - Pause/resume functionality using threading events
    - Async-compatible await_not_paused method
    - Stop with ExecutionStoppedException
    """

    def __init__(self) -> None:
        """Initialize the basic execution controller."""
        self._state = ExecutionState.IDLE
        self._state_lock = threading.RLock()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start in non-paused state
        self._stop_event = threading.Event()

    def pause(self) -> None:
        """Pause the execution at the next checkpoint.

        Raises:
            RuntimeError: If the execution cannot be paused from current state
        """
        with self._state_lock:
            if not self._state.can_pause():
                raise RuntimeError(f"Cannot pause from state {self._state}")

            self._state = ExecutionState.PAUSED
            self._pause_event.clear()
            logger.info("Execution paused")

    def resume(self) -> None:
        """Resume a paused execution.

        Raises:
            RuntimeError: If the execution is not currently paused
        """
        with self._state_lock:
            if not self._state.can_resume():
                raise RuntimeError(f"Cannot resume from state {self._state}")

            self._state = ExecutionState.RUNNING
            self._pause_event.set()
            logger.info("Execution resumed")

    def stop(self) -> None:
        """Stop the execution gracefully.

        Raises:
            RuntimeError: If the execution cannot be stopped from current state
        """
        with self._state_lock:
            if not self._state.can_stop():
                raise RuntimeError(f"Cannot stop from state {self._state}")

            self._state = ExecutionState.STOPPING
            self._stop_event.set()
            # Also set pause event to unblock any waiting threads
            self._pause_event.set()
            logger.info("Execution stopping")

    def start(self) -> None:
        """Start or restart the execution.

        Raises:
            RuntimeError: If the execution cannot be started from current state
        """
        with self._state_lock:
            if not self._state.can_start():
                raise RuntimeError(f"Cannot start from state {self._state}")

            self._state = ExecutionState.RUNNING
            self._pause_event.set()
            self._stop_event.clear()
            logger.info("Execution started")

    def is_paused(self) -> bool:
        """Check if the execution is currently paused.

        Returns:
            True if the execution state is PAUSED
        """
        with self._state_lock:
            return self._state == ExecutionState.PAUSED

    def is_stopped(self) -> bool:
        """Check if the execution has been stopped.

        Returns:
            True if the execution state is STOPPED or STOPPING
        """
        with self._state_lock:
            return self._state.is_terminated()

    def is_running(self) -> bool:
        """Check if the execution is currently running.

        Returns:
            True if the execution state is RUNNING
        """
        with self._state_lock:
            return self._state == ExecutionState.RUNNING

    def get_state(self) -> ExecutionState:
        """Get the current execution state.

        Returns:
            The current ExecutionState
        """
        with self._state_lock:
            return self._state

    def check_pause_point(self) -> None:
        """Check for pause or stop conditions and block if paused.

        This method should be called at regular intervals during execution
        to enable pause/resume functionality. If the execution is paused,
        this method blocks until resumed. If the execution is stopped,
        this method raises ExecutionStoppedException.

        Raises:
            ExecutionStoppedException: If the execution has been stopped
            InterruptedError: If the thread is interrupted while paused
        """
        # Check if stopped
        if self._stop_event.is_set():
            raise ExecutionStoppedException("Execution has been stopped")

        # Block if paused (with timeout to allow checking stop event)
        while not self._pause_event.wait(timeout=0.1):
            # Check if stopped while waiting
            if self._stop_event.is_set():
                raise ExecutionStoppedException("Execution has been stopped")

    async def await_not_paused(self) -> None:
        """Async wait for execution to be not paused.

        This method allows async code to wait for the execution to
        resume from a paused state. It polls the pause state with
        small delays to be async-friendly.

        Raises:
            ExecutionStoppedException: If the execution has been stopped
        """
        while self.is_paused():
            # Check if stopped
            if self._stop_event.is_set():
                raise ExecutionStoppedException("Execution has been stopped")

            # Small async delay before checking again
            await asyncio.sleep(0.1)

    def reset(self) -> None:
        """Reset the controller to IDLE state.

        This method should be called after execution completes or
        when preparing for a new execution cycle.
        """
        with self._state_lock:
            self._state = ExecutionState.IDLE
            self._pause_event.set()
            self._stop_event.clear()
            logger.info("Execution controller reset to IDLE")
