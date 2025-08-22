"""State transition interface - ported from Qontinui framework.

Core interface for state transitions in the model-based GUI automation framework.
"""

from abc import ABC, abstractmethod
from typing import Set, Optional
from enum import Enum


class StaysVisible(Enum):
    """Visibility behavior after transition.
    
    Port of StaysVisible from Qontinui framework enum.
    
    - NONE: Inherit visibility behavior from StateTransitions container
    - TRUE: Source state remains visible after transition  
    - FALSE: Source state becomes hidden after transition
    """
    NONE = "NONE"
    TRUE = "TRUE"
    FALSE = "FALSE"


class StateTransition(ABC):
    """Core interface for state transitions.
    
    Port of StateTransition from Qontinui framework interface.
    
    StateTransition defines the contract for all transition types in the state graph, 
    enabling polymorphic handling of different transition implementations. It represents 
    the edges in the state structure (Ω) that enable navigation between GUI configurations, 
    supporting both programmatic (Python code) and declarative (TaskSequence) transitions.
    
    Transition implementations:
    - TaskSequenceStateTransition: Data-driven transitions using action sequences
    - JavaStateTransition: Code-based transitions using callable functions
    
    Key properties:
    - Activation States: Set of states that this transition activates (navigates to)
    - Exit States: Set of states that are exited when this transition executes
    - Visibility Control: Determines if the source state remains visible after transition
    - Score: Path-finding weight (higher scores discourage using this transition)
    - Success Tracking: Count of successful executions for reliability metrics
    
    Visibility behavior (StaysVisible):
    - NONE: Inherit visibility behavior from StateTransitions container
    - TRUE: Source state remains visible after transition
    - FALSE: Source state becomes hidden after transition
    
    Path-finding integration:
    - Score affects path selection - lower total path scores are preferred
    - Success count can inform reliability-based path selection
    - Multiple activation targets enable branching transitions
    
    In the model-based approach, StateTransition abstracts the mechanism of state 
    navigation while preserving essential metadata for intelligent path selection. The 
    polymorphic design allows mixing declarative and programmatic transitions within 
    the same state graph, providing maximum flexibility for different automation scenarios.
    """
    
    @abstractmethod
    def get_task_sequence_optional(self) -> Optional['TaskSequence']:
        """Get the task sequence if this is a TaskSequence-based transition.
        
        Returns:
            Optional TaskSequence or None
        """
        pass
    
    @abstractmethod
    def get_stays_visible_after_transition(self) -> StaysVisible:
        """Get visibility behavior after transition.
        
        When set, takes precedence over the same variable in StateTransitions.
        Only applies to FromTransitions.
        
        Returns:
            StaysVisible enum value
        """
        pass
    
    @abstractmethod
    def set_stays_visible_after_transition(self, stays_visible: StaysVisible) -> None:
        """Set visibility behavior after transition.
        
        Args:
            stays_visible: StaysVisible enum value
        """
        pass
    
    @abstractmethod
    def get_activate(self) -> Set[int]:
        """Get set of state IDs to activate.
        
        Returns:
            Set of state IDs that this transition activates
        """
        pass
    
    @abstractmethod
    def set_activate(self, activate: Set[int]) -> None:
        """Set states to activate.
        
        Args:
            activate: Set of state IDs to activate
        """
        pass
    
    @abstractmethod
    def get_exit(self) -> Set[int]:
        """Get set of state IDs to exit.
        
        Returns:
            Set of state IDs that are exited
        """
        pass
    
    @abstractmethod
    def set_exit(self, exit: Set[int]) -> None:
        """Set states to exit.
        
        Args:
            exit: Set of state IDs to exit
        """
        pass
    
    @abstractmethod
    def get_score(self) -> int:
        """Get path-finding score.
        
        Larger path scores discourage taking a path with this transition.
        
        Returns:
            Score value
        """
        pass
    
    @abstractmethod
    def set_score(self, score: int) -> None:
        """Set path-finding score.
        
        Args:
            score: Score value
        """
        pass
    
    @abstractmethod
    def get_times_successful(self) -> int:
        """Get count of successful executions.
        
        Returns:
            Number of times this transition executed successfully
        """
        pass
    
    @abstractmethod
    def set_times_successful(self, times_successful: int) -> None:
        """Set count of successful executions.
        
        Args:
            times_successful: Success count
        """
        pass


# Forward reference
class TaskSequence:
    """Placeholder for TaskSequence class."""
    pass