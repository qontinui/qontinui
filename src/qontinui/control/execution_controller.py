"""Execution controller - ported from Qontinui framework.

Interface for controlling execution flow.
"""

from typing import Protocol
from .execution_state import ExecutionState


class ExecutionStoppedException(Exception):
    """Exception thrown when execution has been stopped.
    
    Port of ExecutionStoppedException from Qontinui framework.
    """
    
    def __init__(self, message: str = "Execution has been stopped"):
        """Initialize exception.
        
        Args:
            message: Exception message
        """
        super().__init__(message)


class ExecutionController(Protocol):
    """Interface for controlling the execution flow of automation tasks.
    
    Port of ExecutionController from Qontinui framework interface.
    
    Provides methods to pause, resume, and stop execution, as well as
    query the current execution state.
    
    This interface is the core contract for implementing execution control
    throughout the Qontinui framework. It enables fine-grained control over
    automation execution, allowing for interactive debugging, user intervention,
    and graceful shutdown.
    
    Key features:
    - Thread-safe pause/resume mechanism
    - Graceful stop with cleanup support
    - State query methods for UI integration
    - Checkpoint support for pause points
    
    Implementation requirements:
    - All methods must be thread-safe
    - State transitions must be atomic
    - Pause should block execution until resumed
    - Stop should allow for cleanup operations
    """
    
    def pause(self) -> None:
        """Pause the execution at the next checkpoint.
        
        This method sets the execution state to PAUSED, causing any
        threads checking pause points to block until resume() is called.
        The pause takes effect at the next checkpoint, not immediately.
        
        Raises:
            RuntimeError: If the execution cannot be paused from current state
        """
        ...
    
    def resume(self) -> None:
        """Resume a paused execution.
        
        This method changes the state from PAUSED to RUNNING and notifies
        all waiting threads to continue execution.
        
        Raises:
            RuntimeError: If the execution is not currently paused
        """
        ...
    
    def stop(self) -> None:
        """Stop the execution gracefully.
        
        This method signals all executing threads to stop at the next
        checkpoint. Unlike pause(), stopped executions cannot be resumed.
        
        Raises:
            RuntimeError: If the execution cannot be stopped from current state
        """
        ...
    
    def start(self) -> None:
        """Start or restart the execution.
        
        This method changes the state from IDLE or STOPPED to RUNNING.
        
        Raises:
            RuntimeError: If the execution cannot be started from current state
        """
        ...
    
    def is_paused(self) -> bool:
        """Check if the execution is currently paused.
        
        Returns:
            True if the execution state is PAUSED
        """
        ...
    
    def is_stopped(self) -> bool:
        """Check if the execution has been stopped.
        
        Returns:
            True if the execution state is STOPPED or STOPPING
        """
        ...
    
    def is_running(self) -> bool:
        """Check if the execution is currently running.
        
        Returns:
            True if the execution state is RUNNING
        """
        ...
    
    def get_state(self) -> ExecutionState:
        """Get the current execution state.
        
        Returns:
            The current ExecutionState
        """
        ...
    
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
        ...
    
    def reset(self) -> None:
        """Reset the controller to IDLE state.
        
        This method should be called after execution completes or
        when preparing for a new execution cycle.
        """
        ...