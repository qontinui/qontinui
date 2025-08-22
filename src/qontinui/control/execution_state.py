"""Execution state - ported from Qontinui framework.

Represents the possible states of an automation execution.
"""

from enum import Enum


class ExecutionState(Enum):
    """Represents the possible states of an automation execution.
    
    Port of ExecutionState from Qontinui framework enum.
    
    This enum is used to track and control the execution flow,
    enabling pause, resume, and stop functionality.
    """
    
    IDLE = ("Idle - ready to start",)
    """Execution has not started or has been reset"""
    
    RUNNING = ("Running",)
    """Execution is actively running"""
    
    PAUSED = ("Paused - can be resumed",)
    """Execution has been paused and can be resumed"""
    
    STOPPING = ("Stopping - cleanup in progress",)
    """Execution is in the process of stopping"""
    
    STOPPED = ("Stopped",)
    """Execution has been stopped and cannot be resumed"""
    
    def __init__(self, description: str):
        """Initialize with description.
        
        Args:
            description: Human-readable description of the state
        """
        self.description = description
    
    def get_description(self) -> str:
        """Get the description of this state.
        
        Returns:
            State description string
        """
        return self.description
    
    def is_active(self) -> bool:
        """Check if the execution is in an active state (running or paused).
        
        Returns:
            True if the state is RUNNING or PAUSED
        """
        return self in (ExecutionState.RUNNING, ExecutionState.PAUSED)
    
    def is_terminated(self) -> bool:
        """Check if the execution has terminated (stopped or stopping).
        
        Returns:
            True if the state is STOPPING or STOPPED
        """
        return self in (ExecutionState.STOPPING, ExecutionState.STOPPED)
    
    def can_start(self) -> bool:
        """Check if the execution can be started.
        
        Returns:
            True if the state is IDLE or STOPPED
        """
        return self in (ExecutionState.IDLE, ExecutionState.STOPPED)
    
    def can_pause(self) -> bool:
        """Check if the execution can be paused.
        
        Returns:
            True if the state is RUNNING
        """
        return self == ExecutionState.RUNNING
    
    def can_resume(self) -> bool:
        """Check if the execution can be resumed.
        
        Returns:
            True if the state is PAUSED
        """
        return self == ExecutionState.PAUSED
    
    def can_stop(self) -> bool:
        """Check if the execution can be stopped.
        
        Returns:
            True if the state is RUNNING or PAUSED
        """
        return self in (ExecutionState.RUNNING, ExecutionState.PAUSED)