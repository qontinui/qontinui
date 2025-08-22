"""StateService - ported from Qontinui framework.

Service for managing state operations and navigation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Union
import time
from .state import State
from ..transition.state_transition import StateTransition
from .state_memory import StateMemory
from .path import Path
from .path_finder import PathFinder


@dataclass
class StateService:
    """Service for managing states and navigation.
    
    Port of StateService from Qontinui framework class.
    Provides high-level state management operations.
    """
    
    states: Dict[str, State] = field(default_factory=dict)
    current_state: Optional[State] = None
    state_memory: StateMemory = field(default_factory=StateMemory)
    path_finder: PathFinder = field(default_factory=PathFinder)
    
    # Service configuration
    _max_transition_attempts: int = 3
    _transition_timeout: float = 30.0
    _state_check_interval: float = 0.5
    _auto_navigate: bool = True
    
    # Statistics
    _transitions_executed: int = 0
    _transitions_failed: int = 0
    _states_visited: Set[str] = field(default_factory=set)
    
    def add_state(self, state: State) -> 'StateService':
        """Add a state to the service (fluent).
        
        Args:
            state: State to add
            
        Returns:
            Self for chaining
        """
        self.states[state.name] = state
        self.path_finder.add_state(state)
        return self
    
    def add_states(self, states: List[State]) -> 'StateService':
        """Add multiple states (fluent).
        
        Args:
            states: List of states to add
            
        Returns:
            Self for chaining
        """
        for state in states:
            self.add_state(state)
        return self
    
    def get_state(self, name: str) -> Optional[State]:
        """Get state by name.
        
        Args:
            name: State name
            
        Returns:
            State or None
        """
        return self.states.get(name)
    
    def get_current_state(self) -> Optional[State]:
        """Get the current state.
        
        Returns:
            Current state or None
        """
        # First check if stored current state is still active
        if self.current_state and self.current_state.exists():
            return self.current_state
        
        # Otherwise, find which state is currently active
        for state in self.states.values():
            if state.exists():
                self.set_current_state(state)
                return state
        
        return None
    
    def set_current_state(self, state: State) -> 'StateService':
        """Set the current state (fluent).
        
        Args:
            state: State to set as current
            
        Returns:
            Self for chaining
        """
        self.current_state = state
        self.state_memory.record_state(state)
        self._states_visited.add(state.name)
        return self
    
    def navigate_to(self, target_state: Union[str, State]) -> bool:
        """Navigate to a target state.
        
        Args:
            target_state: Target state or state name
            
        Returns:
            True if navigation succeeded
        """
        # Get target state object
        if isinstance(target_state, str):
            target = self.get_state(target_state)
            if not target:
                return False
        else:
            target = target_state
        
        # Check if already at target
        current = self.get_current_state()
        if current == target:
            return True
        
        # Check if target is directly accessible
        if target.exists():
            self.set_current_state(target)
            return True
        
        # Find path if auto-navigate enabled
        if self._auto_navigate and current:
            path = self.path_finder.find_path(current, target)
            if path:
                return self.execute_path(path)
        
        # Try to reach target directly
        return self.go_to_state(target)
    
    def go_to_state(self, state: State, timeout: Optional[float] = None) -> bool:
        """Go directly to a state.
        
        Args:
            state: Target state
            timeout: Maximum time to wait
            
        Returns:
            True if reached state
        """
        timeout = timeout or self._transition_timeout
        start_time = time.time()
        
        # First check if state already exists
        if state.wait_for(min(5.0, timeout / 2)):
            self.set_current_state(state)
            return True
        
        # Try transitions from current state
        current = self.get_current_state()
        if current:
            transitions = current.get_transitions_to(state)
            for transition in transitions:
                if time.time() - start_time > timeout:
                    break
                
                if self.execute_transition(transition):
                    if state.wait_for(min(5.0, timeout - (time.time() - start_time))):
                        self.set_current_state(state)
                        return True
        
        return False
    
    def execute_transition(self, transition: StateTransition) -> bool:
        """Execute a single transition.
        
        Args:
            transition: Transition to execute
            
        Returns:
            True if transition succeeded
        """
        attempts = 0
        while attempts < self._max_transition_attempts:
            attempts += 1
            
            try:
                if transition.execute():
                    self._transitions_executed += 1
                    self.state_memory.record_transition(transition)
                    
                    # Update current state if transition succeeded
                    if transition.to_state:
                        if transition.to_state.wait_for(5.0):
                            self.set_current_state(transition.to_state)
                    
                    return True
            except Exception as e:
                print(f"Transition failed: {e}")
            
            if attempts < self._max_transition_attempts:
                time.sleep(1.0)  # Wait before retry
        
        self._transitions_failed += 1
        return False
    
    def execute_path(self, path: Path) -> bool:
        """Execute a path of states.
        
        Args:
            path: Path to execute
            
        Returns:
            True if reached end of path
        """
        for i, state in enumerate(path.states):
            # Check if already at state
            if state.exists():
                self.set_current_state(state)
                continue
            
            # Get transition to next state
            if i > 0:
                prev_state = path.states[i - 1]
                transition = path.get_transition(i - 1)
                
                if transition:
                    if not self.execute_transition(transition):
                        return False
                else:
                    # No specific transition, try to navigate
                    if not self.go_to_state(state):
                        return False
            else:
                # First state in path
                if not self.go_to_state(state):
                    return False
        
        return True
    
    def find_state(self, timeout: float = 30.0) -> Optional[State]:
        """Find which state is currently active.
        
        Args:
            timeout: Maximum search time
            
        Returns:
            Active state or None
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            for state in self.states.values():
                if state.exists():
                    self.set_current_state(state)
                    return state
            
            time.sleep(self._state_check_interval)
        
        return None
    
    def wait_for_state(self, state: State, timeout: float = 30.0) -> bool:
        """Wait for a specific state to appear.
        
        Args:
            state: State to wait for
            timeout: Maximum wait time
            
        Returns:
            True if state appeared
        """
        if state.wait_for(timeout):
            self.set_current_state(state)
            return True
        return False
    
    def wait_for_any_state(self, states: List[State], 
                          timeout: float = 30.0) -> Optional[State]:
        """Wait for any of the specified states.
        
        Args:
            states: List of states to wait for
            timeout: Maximum wait time
            
        Returns:
            First state that appears or None
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            for state in states:
                if state.exists():
                    self.set_current_state(state)
                    return state
            
            time.sleep(self._state_check_interval)
        
        return None
    
    def get_possible_states(self) -> List[State]:
        """Get all states reachable from current state.
        
        Returns:
            List of reachable states
        """
        current = self.get_current_state()
        if not current:
            return []
        
        return current.get_possible_next_states()
    
    def get_state_history(self) -> List[State]:
        """Get history of visited states.
        
        Returns:
            List of states in visit order
        """
        return self.state_memory.get_state_history()
    
    def get_transition_history(self) -> List[StateTransition]:
        """Get history of executed transitions.
        
        Returns:
            List of transitions in execution order
        """
        return self.state_memory.get_transition_history()
    
    def reset(self) -> 'StateService':
        """Reset service state (fluent).
        
        Returns:
            Self for chaining
        """
        self.current_state = None
        self.state_memory.clear()
        self._transitions_executed = 0
        self._transitions_failed = 0
        self._states_visited.clear()
        return self
    
    def set_auto_navigate(self, enable: bool = True) -> 'StateService':
        """Enable/disable auto-navigation (fluent).
        
        Args:
            enable: True to enable auto-navigation
            
        Returns:
            Self for chaining
        """
        self._auto_navigate = enable
        return self
    
    def set_max_attempts(self, attempts: int) -> 'StateService':
        """Set maximum transition attempts (fluent).
        
        Args:
            attempts: Maximum attempts
            
        Returns:
            Self for chaining
        """
        self._max_transition_attempts = attempts
        return self
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'states_visited': len(self._states_visited),
            'transitions_executed': self._transitions_executed,
            'transitions_failed': self._transitions_failed,
            'success_rate': (self._transitions_executed / 
                           max(1, self._transitions_executed + self._transitions_failed)),
            'current_state': self.current_state.name if self.current_state else None,
            'total_states': len(self.states)
        }
    
    def __str__(self) -> str:
        """String representation."""
        current = self.current_state.name if self.current_state else "None"
        return f"StateService(states={len(self.states)}, current='{current}')"