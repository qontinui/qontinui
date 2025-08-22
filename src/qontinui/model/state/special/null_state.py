"""NullState class - ported from Qontinui framework.

Special state for handling stateless objects in the framework.
"""

from enum import Enum
from ..state import State
from ..state_enum import StateEnum


class NullStateName(Enum):
    """Enum for NullState.
    
    Port of NullState from Qontinui framework.Name enum.
    Implements StateEnum interface.
    """
    NULL = "NULL"
    
    @property
    def name(self) -> str:
        """Get state name."""
        return self.value


class NullState:
    """Special state for handling stateless objects in the framework.
    
    Port of NullState from Qontinui framework class.
    
    NullState provides a container for objects that don't belong to any specific state 
    in the application's state graph. It enables the framework to process temporary or 
    standalone objects using the same action infrastructure designed for state-based 
    automation, maintaining consistency in how all objects are handled.
    
    Key characteristics:
    - Stateless Context: Objects belong to no particular application state
    - No State Activation: Finding these objects doesn't trigger state changes
    - Temporary Objects: Typically used for transient elements or utilities
    - Repository Exclusion: Should not be stored in the state repository
    - Action Compatibility: Can be acted upon like any other state object
    
    Common use cases:
    - Processing temporary dialogs or notifications
    - Handling utility objects that appear across multiple states
    - Working with objects during state transitions
    - Testing individual patterns without state context
    - Operating on objects before state structure is established
    
    Design pattern benefits:
    - Enables uniform object handling regardless of state association
    - Simplifies action implementation by avoiding null checks
    - Provides clear semantic meaning for stateless operations
    - Maintains type safety in the state system
    
    Implementation notes:
    - Contains a single State instance named "null"
    - Implements StateEnum through the Name enum for type compatibility
    - Should be used sparingly - most objects should belong to states
    - Not a singleton to allow multiple contexts if needed
    """
    
    def __init__(self):
        """Initialize NullState with a null state."""
        self._state = State(name="null", state_enum=NullStateName.NULL)
    
    @property
    def state(self) -> State:
        """Get the null state.
        
        Returns:
            The null State instance
        """
        return self._state
    
    @property
    def name(self) -> NullStateName:
        """Get the state name enum.
        
        Returns:
            NullStateName.NULL
        """
        return NullStateName.NULL
    
    def is_null(self) -> bool:
        """Check if this is a null state.
        
        Returns:
            Always True for NullState
        """
        return True
    
    def __str__(self) -> str:
        """String representation."""
        return "NullState"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"NullState(state={self._state.name})"
    
    @classmethod
    def instance(cls) -> 'NullState':
        """Get a NullState instance.
        
        Note: Not a singleton to allow multiple contexts if needed.
        
        Returns:
            New NullState instance
        """
        return cls()