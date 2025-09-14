"""Initial states module - ported from Qontinui framework.

Manages the collection of initial states that are available when the application starts.
"""

from typing import Set, Dict, Optional, List
from dataclasses import dataclass, field
import logging

from .state import State

logger = logging.getLogger(__name__)


@dataclass
class InitialStates:
    """Manages the collection of initial states available at application startup.
    
    Port of InitialStates from Qontinui framework.
    
    This class maintains a registry of states that should be available when the
    application starts. These are typically the entry points into the application's
    state machine, such as login screens, main menus, or splash screens.
    
    The InitialStates serves several purposes:
    - Provides a starting point for state navigation
    - Ensures critical states are always registered
    - Helps with state discovery and recovery
    - Facilitates testing by providing known entry points
    """
    
    _states: Dict[str, State] = field(default_factory=dict)
    """Registry of initial states by name."""
    
    _state_names: Set[str] = field(default_factory=set)
    """Set of registered state names for quick lookup."""
    
    _default_state: Optional[str] = None
    """Name of the default initial state."""
    
    def add_state(self, state: State) -> None:
        """Add a state to the initial states registry.
        
        Args:
            state: State to add
        """
        if not state.name:
            logger.warning("Cannot add state without name to initial states")
            return
            
        self._states[state.name] = state
        self._state_names.add(state.name)
        logger.debug(f"Added initial state: {state.name}")
        
        # Set as default if it's the first state
        if self._default_state is None:
            self._default_state = state.name
            logger.debug(f"Set default initial state: {state.name}")
    
    def add_states(self, *states: State) -> None:
        """Add multiple states to the initial states registry.
        
        Args:
            *states: States to add
        """
        for state in states:
            self.add_state(state)
    
    def get_state(self, name: str) -> Optional[State]:
        """Get a state by name.
        
        Args:
            name: Name of the state
            
        Returns:
            State if found, None otherwise
        """
        return self._states.get(name)
    
    def get_all_states(self) -> List[State]:
        """Get all registered initial states.
        
        Returns:
            List of all initial states
        """
        return list(self._states.values())
    
    def get_state_names(self) -> Set[str]:
        """Get all registered state names.
        
        Returns:
            Set of state names
        """
        return self._state_names.copy()
    
    def has_state(self, name: str) -> bool:
        """Check if a state is registered.
        
        Args:
            name: Name of the state
            
        Returns:
            True if state is registered
        """
        return name in self._state_names
    
    def set_default_state(self, name: str) -> bool:
        """Set the default initial state.
        
        Args:
            name: Name of the state to set as default
            
        Returns:
            True if successful, False if state not found
        """
        if name not in self._state_names:
            logger.warning(f"Cannot set default state - '{name}' not found in initial states")
            return False
            
        self._default_state = name
        logger.debug(f"Set default initial state: {name}")
        return True
    
    def get_default_state(self) -> Optional[State]:
        """Get the default initial state.
        
        Returns:
            Default state if set, None otherwise
        """
        if self._default_state:
            return self._states.get(self._default_state)
        return None
    
    def get_default_state_name(self) -> Optional[str]:
        """Get the name of the default initial state.
        
        Returns:
            Default state name if set, None otherwise
        """
        return self._default_state
    
    def remove_state(self, name: str) -> bool:
        """Remove a state from the registry.
        
        Args:
            name: Name of the state to remove
            
        Returns:
            True if removed, False if not found
        """
        if name not in self._state_names:
            return False
            
        del self._states[name]
        self._state_names.remove(name)
        
        # Clear default if it was removed
        if self._default_state == name:
            self._default_state = None
            # Set a new default if there are other states
            if self._state_names:
                self._default_state = next(iter(self._state_names))
                logger.debug(f"Set new default initial state: {self._default_state}")
        
        logger.debug(f"Removed initial state: {name}")
        return True
    
    def clear(self) -> None:
        """Clear all initial states."""
        self._states.clear()
        self._state_names.clear()
        self._default_state = None
        logger.debug("Cleared all initial states")
    
    def is_empty(self) -> bool:
        """Check if there are no initial states.
        
        Returns:
            True if no states registered
        """
        return len(self._states) == 0
    
    def size(self) -> int:
        """Get the number of registered initial states.
        
        Returns:
            Number of states
        """
        return len(self._states)
    
    def __str__(self) -> str:
        """String representation."""
        default_marker = f" (default: {self._default_state})" if self._default_state else ""
        return f"InitialStates[{self.size()} states{default_marker}]"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        states_list = ', '.join(sorted(self._state_names))
        return f"InitialStates(states=[{states_list}], default={self._default_state})"


# Global singleton instance
_initial_states = InitialStates()


def get_initial_states() -> InitialStates:
    """Get the global InitialStates instance.
    
    Returns:
        The global InitialStates singleton
    """
    return _initial_states


def register_initial_state(state: State) -> None:
    """Convenience function to register an initial state.
    
    Args:
        state: State to register
    """
    _initial_states.add_state(state)


def register_initial_states(*states: State) -> None:
    """Convenience function to register multiple initial states.
    
    Args:
        *states: States to register
    """
    _initial_states.add_states(*states)