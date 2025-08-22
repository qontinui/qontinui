"""StateMemory - ported from Qontinui framework.

Runtime tracker of currently active states.
"""

from dataclasses import dataclass, field
from typing import Set, List, Optional, Dict
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


class SpecialStateId(IntEnum):
    """Special state IDs for state management.
    
    Port of special from Qontinui framework state enums.
    """
    UNKNOWN = -1
    PREVIOUS = -2
    CURRENT = -3
    EXPECTED = -4
    NULL = -5


@dataclass
class StateMemory:
    """Runtime tracker of currently active states.
    
    Port of StateMemory from Qontinui framework class.
    
    StateMemory maintains the set of currently active states in the application.
    It tracks which states are visible/active at any given moment and updates
    this information based on visual matches found during action execution.
    
    This is NOT a history tracker - it only maintains current state information.
    """
    
    # Currently active state IDs
    active_states: Set[int] = field(default_factory=set)
    
    # Special state collections
    previous_states: Set[int] = field(default_factory=set)
    expected_states: Set[int] = field(default_factory=set)
    
    def get_active_state_list(self) -> List[int]:
        """Get list of currently active state IDs.
        
        Returns:
            List of active state IDs
        """
        return list(self.active_states)
    
    def get_active_state_names(self, state_service: 'StateService') -> List[str]:
        """Get names of currently active states.
        
        Args:
            state_service: Service to resolve state names
            
        Returns:
            List of active state names
        """
        names = []
        for state_id in self.active_states:
            name = state_service.get_state_name(state_id)
            if name:
                names.append(name)
        return names
    
    def add_active_state(self, state_id: int, state: Optional['State'] = None) -> None:
        """Mark a state as active.
        
        When a state becomes active:
        - Add to active states set
        - Set probability to 100%
        - Increment times visited
        
        Args:
            state_id: ID of state to activate
            state: Optional state object to update
        """
        # Add to active set
        self.active_states.add(state_id)
        
        # Update state object if provided
        if state:
            state.probability_exists = 100.0
            state.times_visited += 1
            
        logger.debug(f"State {state_id} activated. Active states: {self.active_states}")
    
    def remove_inactive_state(self, state_id: int, state: Optional['State'] = None) -> None:
        """Mark a state as inactive.
        
        When a state becomes inactive:
        - Remove from active states set
        - Set probability to 0%
        
        Args:
            state_id: ID of state to deactivate
            state: Optional state object to update
        """
        # Remove from active set
        self.active_states.discard(state_id)
        
        # Update state object if provided
        if state:
            state.probability_exists = 0.0
            
        logger.debug(f"State {state_id} deactivated. Active states: {self.active_states}")
    
    def adjust_active_states_with_matches(self, matches: 'Matches', 
                                         state_service: 'StateService') -> None:
        """Update active states based on visual matches found.
        
        This is called after Find operations to update which states
        are currently active based on what was found on screen.
        
        Args:
            matches: Matches found during action
            state_service: Service to resolve states
        """
        # Track states found in matches
        found_state_ids = set()
        
        # Check each match for state associations
        for match in matches.get_matches():
            if hasattr(match, 'state_object') and match.state_object:
                # Get owner state from the matched object
                if hasattr(match.state_object, 'owner_state_id'):
                    state_id = match.state_object.owner_state_id
                    if state_id and state_id > 0:  # Valid state ID
                        found_state_ids.add(state_id)
        
        # Add newly found states
        for state_id in found_state_ids:
            if state_id not in self.active_states:
                state = state_service.get_state(state_id)
                self.add_active_state(state_id, state)
        
        # Optionally remove states that weren't found
        # (This depends on the search scope and configuration)
        logger.debug(f"Adjusted active states from matches. Found states: {found_state_ids}")
    
    def set_active_states(self, state_ids: Set[int]) -> None:
        """Set the complete set of active states.
        
        Replaces all current active states.
        
        Args:
            state_ids: New set of active state IDs
        """
        self.previous_states = self.active_states.copy()
        self.active_states = state_ids.copy()
        logger.debug(f"Active states set to: {self.active_states}")
    
    def clear_active_states(self) -> None:
        """Remove all active states."""
        self.previous_states = self.active_states.copy()
        self.active_states.clear()
        logger.debug("All active states cleared")
    
    def is_active(self, state_id: int) -> bool:
        """Check if a state is currently active.
        
        Args:
            state_id: State ID to check
            
        Returns:
            True if state is active
        """
        return state_id in self.active_states
    
    def has_active_states(self) -> bool:
        """Check if any states are active.
        
        Returns:
            True if at least one state is active
        """
        return len(self.active_states) > 0
    
    def get_active_state_count(self) -> int:
        """Get number of active states.
        
        Returns:
            Count of active states
        """
        return len(self.active_states)
    
    def save_as_previous(self) -> None:
        """Save current active states as previous."""
        self.previous_states = self.active_states.copy()
    
    def restore_previous(self) -> None:
        """Restore previous active states."""
        self.active_states = self.previous_states.copy()
    
    def set_expected_states(self, state_ids: Set[int]) -> None:
        """Set expected future states.
        
        Used for predictive state management.
        
        Args:
            state_ids: Expected state IDs
        """
        self.expected_states = state_ids.copy()
    
    def is_expected(self, state_id: int) -> bool:
        """Check if a state is expected.
        
        Args:
            state_id: State ID to check
            
        Returns:
            True if state is expected
        """
        return state_id in self.expected_states
    
    def remove_all_states(self) -> None:
        """Complete reset of state memory."""
        self.active_states.clear()
        self.previous_states.clear()
        self.expected_states.clear()
        logger.debug("StateMemory completely reset")
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            Description of active states
        """
        if not self.active_states:
            return "StateMemory(no active states)"
        return f"StateMemory(active={self.active_states})"