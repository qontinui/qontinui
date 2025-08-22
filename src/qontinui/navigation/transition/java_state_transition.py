"""Java state transition - ported from Qontinui framework.

Code-based state transition implementation.
"""

from typing import Set, Optional, Callable, List
from dataclasses import dataclass, field

from ...model.transition.state_transition import StateTransition, StaysVisible


@dataclass
class JavaStateTransition(StateTransition):
    """Code-based state transition implementation.
    
    Port of JavaStateTransition from Qontinui framework class.
    
    JavaStateTransition represents transitions defined through Python code using callable 
    functions. This implementation enables dynamic, programmatic state navigation where transition 
    logic can involve complex conditions, external data, or runtime calculations that cannot be 
    expressed declaratively.
    
    Key components:
    - Transition Function: Callable that executes the transition logic
    - Activation List: States to activate after successful transition
    - Exit List: States to deactivate after successful transition
    - Visibility Control: Whether source state remains visible post-transition
    - Path Score: Weight for path-finding algorithms (higher = less preferred)
    
    State reference handling:
    - Uses state names during definition (IDs not yet assigned)
    - Names are converted to IDs during initialization
    - Both name and ID sets are maintained for flexibility
    - Supports multiple target states for branching transitions
    
    Transition execution flow:
    1. Callable is invoked to perform transition logic
    2. If True is returned, transition is considered successful
    3. States in 'activate' set become active
    4. States in 'exit' set become inactive
    5. Success counter is incremented for metrics
    
    Common use patterns:
    - Complex navigation logic that depends on runtime conditions
    - Transitions involving external API calls or data validation
    - Dynamic state activation based on application state
    - Fallback transitions with custom error handling
    
    In the model-based approach, JavaStateTransition provides the flexibility needed for 
    complex automation scenarios where declarative TaskSequences are insufficient. It 
    enables seamless integration of custom logic while maintaining the benefits of the 
    state graph structure for navigation and path finding.
    """
    
    type: str = "java"
    """Transition type identifier."""
    
    transition_function: Optional[Callable[[], bool]] = None
    """Function containing transition logic."""
    
    stays_visible_after_transition: StaysVisible = StaysVisible.NONE
    """Visibility behavior after transition."""
    
    activate_names: Set[str] = field(default_factory=set)
    """State names to activate (used during definition)."""
    
    exit_names: Set[str] = field(default_factory=set)
    """State names to exit (used during definition)."""
    
    activate: Set[int] = field(default_factory=set)
    """State IDs to activate (used at runtime)."""
    
    exit: Set[int] = field(default_factory=set)
    """State IDs to exit (used at runtime)."""
    
    score: int = 0
    """Path-finding score (higher = less preferred)."""
    
    times_successful: int = 0
    """Count of successful executions."""
    
    def get_task_sequence_optional(self) -> Optional['TaskSequence']:
        """Get task sequence (always None for JavaStateTransition).
        
        Returns:
            None - JavaStateTransition uses functions, not TaskSequences
        """
        return None
    
    def get_stays_visible_after_transition(self) -> StaysVisible:
        """Get visibility behavior after transition.
        
        Returns:
            StaysVisible enum value
        """
        return self.stays_visible_after_transition
    
    def set_stays_visible_after_transition(self, stays_visible: StaysVisible) -> None:
        """Set visibility behavior after transition.
        
        Args:
            stays_visible: StaysVisible enum value
        """
        self.stays_visible_after_transition = stays_visible
    
    def get_activate(self) -> Set[int]:
        """Get set of state IDs to activate.
        
        Returns:
            Set of state IDs
        """
        return self.activate
    
    def set_activate(self, activate: Set[int]) -> None:
        """Set states to activate.
        
        Args:
            activate: Set of state IDs
        """
        self.activate = activate
    
    def get_exit(self) -> Set[int]:
        """Get set of state IDs to exit.
        
        Returns:
            Set of state IDs
        """
        return self.exit
    
    def set_exit(self, exit: Set[int]) -> None:
        """Set states to exit.
        
        Args:
            exit: Set of state IDs
        """
        self.exit = exit
    
    def get_score(self) -> int:
        """Get path-finding score.
        
        Returns:
            Score value
        """
        return self.score
    
    def set_score(self, score: int) -> None:
        """Set path-finding score.
        
        Args:
            score: Score value
        """
        self.score = score
    
    def get_times_successful(self) -> int:
        """Get count of successful executions.
        
        Returns:
            Success count
        """
        return self.times_successful
    
    def set_times_successful(self, times_successful: int) -> None:
        """Set count of successful executions.
        
        Args:
            times_successful: Success count
        """
        self.times_successful = times_successful
    
    def __str__(self) -> str:
        """String representation for debugging."""
        return f"activate={self.activate_names}, exit={self.exit_names}"


class JavaStateTransitionBuilder:
    """Builder for creating JavaStateTransition instances fluently.
    
    Port of JavaStateTransition from Qontinui framework.Builder class.
    
    Provides a readable API for constructing transitions with sensible defaults:
    - Default function returns False (transition fails)
    - Default visibility is NONE (inherit from transition type)
    - Empty activation and exit sets by default
    - Zero score (highest priority)
    
    Example usage:
        transition = JavaStateTransitionBuilder()\\
            .set_function(lambda: action.click("Next"))\\
            .add_to_activate("PageTwo")\\
            .add_to_exit("PageOne")\\
            .set_score(10)\\
            .build()
    """
    
    def __init__(self):
        """Initialize builder with defaults."""
        self.transition_function = lambda: False
        self.stays_visible_after_transition = StaysVisible.NONE
        self.activate = set()
        self.exit = set()
        self.score = 0
    
    def set_function(self, function: Callable[[], bool]) -> 'JavaStateTransitionBuilder':
        """Set the transition logic function.
        
        The callable should contain all logic needed to execute
        the transition, returning True on success and False on failure.
        
        Args:
            function: Function containing transition logic
            
        Returns:
            This builder for method chaining
        """
        self.transition_function = function
        return self
    
    def set_stays_visible_after_transition(self, stays_visible) -> 'JavaStateTransitionBuilder':
        """Set visibility behavior.
        
        Controls whether the source state remains visible after transition:
        - TRUE: Source state stays active
        - FALSE: Source state is deactivated  
        - NONE: Behavior determined by transition type
        
        Args:
            stays_visible: StaysVisible enum or boolean
            
        Returns:
            This builder for method chaining
        """
        if isinstance(stays_visible, bool):
            self.stays_visible_after_transition = StaysVisible.TRUE if stays_visible else StaysVisible.FALSE
        else:
            self.stays_visible_after_transition = stays_visible
        return self
    
    def add_to_activate(self, *state_names: str) -> 'JavaStateTransitionBuilder':
        """Add states to activate after successful transition.
        
        These states will become active when the transition succeeds.
        Can be called multiple times to accumulate states.
        
        Args:
            state_names: Variable number of state names to activate
            
        Returns:
            This builder for method chaining
        """
        self.activate.update(state_names)
        return self
    
    def add_to_exit(self, *state_names: str) -> 'JavaStateTransitionBuilder':
        """Add states to exit after successful transition.
        
        These states will be deactivated when the transition succeeds.
        Can be called multiple times to accumulate states.
        
        Args:
            state_names: Variable number of state names to exit
            
        Returns:
            This builder for method chaining
        """
        self.exit.update(state_names)
        return self
    
    def set_score(self, score: int) -> 'JavaStateTransitionBuilder':
        """Set the path-finding score for this transition.
        
        Higher scores make this transition less preferred when multiple
        paths exist. Score 0 is highest priority.
        
        Args:
            score: Path-finding weight (0 = best)
            
        Returns:
            This builder for method chaining
        """
        self.score = score
        return self
    
    def build(self) -> JavaStateTransition:
        """Create the JavaStateTransition with configured properties.
        
        Constructs a new instance with all builder settings applied.
        The builder can be reused after calling build().
        
        Returns:
            Configured JavaStateTransition instance
        """
        transition = JavaStateTransition()
        transition.transition_function = self.transition_function
        transition.stays_visible_after_transition = self.stays_visible_after_transition
        transition.activate_names = self.activate.copy()
        transition.exit_names = self.exit.copy()
        transition.score = self.score
        return transition


# Forward reference
class TaskSequence:
    """Placeholder for TaskSequence class."""
    pass