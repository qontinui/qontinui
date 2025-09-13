"""Scheduled task definition for state-aware scheduling.

Tasks that can be executed based on application state.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Set, Any, Dict
from enum import Enum
from datetime import datetime


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0  # Highest priority
    HIGH = 1
    NORMAL = 2
    LOW = 3
    IDLE = 4  # Lowest priority


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"  # Blocked by state requirements


@dataclass
class ScheduledTask:
    """A task that can be scheduled based on state.
    
    Following Brobot principles:
    - Tasks have state requirements
    - Tasks can transition to target states
    - Priority-based scheduling
    - Can be recurring or one-time
    """
    
    # Task identification
    name: str
    description: str = ""
    
    # Task function
    task_function: Optional[Callable[[], bool]] = None
    """Function to execute. Returns True on success."""
    
    # State requirements
    required_states: Set[str] = field(default_factory=set)
    """States that must be active for task to run."""
    
    forbidden_states: Set[str] = field(default_factory=set)
    """States that must NOT be active for task to run."""
    
    target_state: Optional[str] = None
    """State to transition to when executing this task."""
    
    # Scheduling
    priority: TaskPriority = TaskPriority.NORMAL
    recurring: bool = False
    max_retries: int = 3
    retry_count: int = 0
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Status
    status: TaskStatus = TaskStatus.PENDING
    last_error: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def can_run_in_states(self, active_states: Set[str]) -> bool:
        """Check if task can run with given active states.
        
        Args:
            active_states: Currently active state names
            
        Returns:
            True if task can run
        """
        # Check required states are active
        if self.required_states:
            if not self.required_states.issubset(active_states):
                return False
        
        # Check forbidden states are not active
        if self.forbidden_states:
            if self.forbidden_states.intersection(active_states):
                return False
        
        return True
    
    def is_ready(self, active_states: Set[str]) -> bool:
        """Check if task is ready to execute.
        
        Args:
            active_states: Currently active state names
            
        Returns:
            True if ready to execute
        """
        # Must be in pending or ready status
        if self.status not in [TaskStatus.PENDING, TaskStatus.READY]:
            return False
        
        # Check state requirements
        return self.can_run_in_states(active_states)
    
    def execute(self) -> bool:
        """Execute the task.
        
        Returns:
            True if successful
        """
        if not self.task_function:
            self.last_error = "No task function defined"
            return False
        
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()
        
        try:
            success = self.task_function()
            
            if success:
                self.status = TaskStatus.COMPLETED
                self.completed_at = datetime.now()
                self.retry_count = 0
            else:
                self.status = TaskStatus.FAILED
                self.retry_count += 1
                self.last_error = "Task function returned False"
            
            return success
            
        except Exception as e:
            self.status = TaskStatus.FAILED
            self.retry_count += 1
            self.last_error = str(e)
            return False
    
    def reset(self):
        """Reset task for re-execution (for recurring tasks)."""
        self.status = TaskStatus.PENDING
        self.started_at = None
        self.completed_at = None
        self.last_error = None
        # Don't reset retry count for recurring tasks
    
    def should_retry(self) -> bool:
        """Check if task should be retried.
        
        Returns:
            True if should retry
        """
        return (
            self.status == TaskStatus.FAILED and
            self.retry_count < self.max_retries
        )
    
    def __lt__(self, other: 'ScheduledTask') -> bool:
        """Compare tasks by priority (for heap queue).
        
        Args:
            other: Other task to compare
            
        Returns:
            True if this task has higher priority (lower value)
        """
        return self.priority.value < other.priority.value