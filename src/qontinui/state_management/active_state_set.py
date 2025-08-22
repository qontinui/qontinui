"""Active state set management - ported from Qontinui framework.

Manages a collection of currently active states in the framework.
"""

from typing import Set
from ..model.state.state_enum import StateEnum


class ActiveStateSet:
    """Manages a collection of currently active states.
    
    Port of ActiveStateSet from Qontinui framework class.
    
    ActiveStateSet provides a lightweight container for tracking which states are currently 
    active in the GUI using StateEnum identifiers. This class is particularly useful during 
    the development phase when working with enum-based state definitions, before transitioning 
    to the more robust database-backed state management system.
    
    Key features:
    - Set Semantics: Ensures each state is tracked only once, preventing duplicates
    - Bulk Operations: Supports adding individual states or entire collections
    - Enum-based: Works with StateEnum interface for compile-time type safety
    - Merge Support: Can combine multiple ActiveStateSet instances
    
    Use cases:
    - Tracking active states during automation execution
    - Building state sets for transition operations
    - Maintaining state context in enum-based automation scripts
    - Debugging state transitions by examining active state sets
    
    This class represents a simpler alternative to StateMemory for scenarios where 
    database integration is not required. It's particularly useful for:
    - Unit testing state management logic
    - Lightweight automation scripts
    - Prototyping state structures before database implementation
    
    In the model-based approach, ActiveStateSet provides a foundation for tracking the 
    current GUI configuration without the overhead of full state management infrastructure. 
    This makes it ideal for simpler automation scenarios or as a stepping stone to more 
    sophisticated state tracking mechanisms.
    """
    
    def __init__(self):
        """Initialize empty active state set."""
        self.active_states: Set[StateEnum] = set()
    
    def add_state(self, state_enum: StateEnum) -> None:
        """Add a single state to the active set.
        
        Args:
            state_enum: State to add
        """
        self.active_states.add(state_enum)
    
    def add_states(self, states: Set[StateEnum]) -> None:
        """Add multiple states to the active set.
        
        Args:
            states: Set of states to add
        """
        self.active_states.update(states)
    
    def add_states_from_set(self, active_states: 'ActiveStateSet') -> None:
        """Add states from another ActiveStateSet.
        
        Args:
            active_states: ActiveStateSet to merge
        """
        self.active_states.update(active_states.get_active_states())
    
    def get_active_states(self) -> Set[StateEnum]:
        """Get the set of active states.
        
        Returns:
            Set of active state enums
        """
        return self.active_states
    
    def clear(self) -> None:
        """Clear all active states."""
        self.active_states.clear()
    
    def remove_state(self, state_enum: StateEnum) -> None:
        """Remove a state from the active set.
        
        Args:
            state_enum: State to remove
        """
        self.active_states.discard(state_enum)
    
    def is_active(self, state_enum: StateEnum) -> bool:
        """Check if a state is active.
        
        Args:
            state_enum: State to check
            
        Returns:
            True if state is active
        """
        return state_enum in self.active_states
    
    def __len__(self) -> int:
        """Get count of active states."""
        return len(self.active_states)
    
    def __str__(self) -> str:
        """String representation."""
        state_names = [str(s) for s in self.active_states]
        return f"ActiveStateSet({', '.join(state_names)})"