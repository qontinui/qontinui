"""State memory management - ported from Qontinui framework.

Maintains the runtime memory of active States in the framework.
"""

from typing import Set, List, Optional
from enum import Enum
import logging

from ..model.state.state import State
from ..actions.action_result import ActionResult


logger = logging.getLogger(__name__)


class StateMemoryEnum(Enum):
    """Special state references.
    
    Port of StateMemory from Qontinui framework.Enum.
    """
    
    PREVIOUS = "PREVIOUS"
    """References states that are currently hidden but can be returned to."""
    
    CURRENT = "CURRENT"
    """The set of currently active states."""
    
    EXPECTED = "EXPECTED"
    """States anticipated to become active after transitions."""


class StateMemory:
    """Maintains the runtime memory of active States.
    
    Port of StateMemory from Qontinui framework class.
    
    StateMemory is a critical component of the State Management System (μ), responsible 
    for tracking which states are currently active in the GUI. It serves as the framework's 
    working memory, maintaining an accurate understanding of the current GUI configuration 
    throughout automation execution.
    
    Key responsibilities:
    - Active State Tracking: Maintains a set of currently visible/active states
    - State Transitions: Updates active states as the GUI changes
    - Match Integration: Adjusts active states based on what is found during Find operations
    - State Probability: Manages probability values for mock testing and uncertainty handling
    - Visit Tracking: Records state visit counts for analysis and optimization
    
    Special state handling:
    - PREVIOUS: References states that are currently hidden but can be returned to
    - CURRENT: The set of currently active states
    - EXPECTED: States anticipated to become active after transitions
    - NULL State: Ignored in active state tracking as it represents stateless elements
    
    In the model-based approach, StateMemory bridges the gap between the static state 
    structure and the dynamic runtime behavior of the GUI. It enables the framework to 
    maintain context awareness, recover from unexpected situations, and make intelligent 
    decisions about navigation and action execution.
    """
    
    def __init__(self, state_service: Optional['StateService'] = None):
        """Initialize state memory.
        
        Args:
            state_service: Service for accessing state definitions
        """
        self.state_service = state_service
        self.active_states: Set[int] = set()
        """Set of active state IDs."""
    
    def get_active_state_list(self) -> List[State]:
        """Retrieve all active states as State objects.
        
        Converts the internal set of active state IDs to a list of State objects
        by looking up each ID in the StateService. Invalid IDs are silently skipped.
        
        Returns:
            List of currently active State objects, empty if no states active
        """
        if not self.state_service:
            return []
        
        active_state_list = []
        for state_id in self.active_states:
            state = self.state_service.get_state(state_id)
            if state:
                active_state_list.append(state)
        return active_state_list
    
    def get_active_state_names(self) -> List[str]:
        """Retrieve names of all active states.
        
        Provides a human-readable list of active state names for debugging,
        logging, and display purposes. Names are extracted from State objects
        via StateService lookup.
        
        Returns:
            List of active state names, empty if no states active
        """
        if not self.state_service:
            return []
        
        active_state_names = []
        for state_id in self.active_states:
            state = self.state_service.get_state(state_id)
            if state:
                active_state_names.append(state.name)
        return active_state_names
    
    def get_active_state_names_as_string(self) -> str:
        """Format active state names as a comma-separated string.
        
        Convenience method for displaying active states in logs and reports.
        Useful for concise status messages and debugging output.
        
        Returns:
            Comma-separated string of active state names
        """
        return ", ".join(self.get_active_state_names())
    
    def adjust_active_states_with_matches(self, matches: ActionResult) -> None:
        """Update active states based on matches found during Find operations.
        
        When the Find action discovers state objects on screen, this method
        ensures their owning states are marked as active. This automatic
        state discovery helps maintain accurate state tracking even when
        states change unexpectedly.
        
        Side effects:
        - Adds states to active list based on found matches
        - Updates state probabilities and visit counts
        - Ignores matches without valid state ownership data
        
        Args:
            matches: ActionResult containing found state objects
        """
        for match in matches.get_match_list():
            if match.state_object_data:
                owner_state_id = match.state_object_data.owner_state_id
                if owner_state_id and owner_state_id > 0:
                    self.add_active_state(owner_state_id)
    
    def add_active_state(self, state_id: int) -> None:
        """Add a state to the active state list.
        
        Args:
            state_id: ID of state to activate
        """
        if state_id and state_id > 0:
            self.active_states.add(state_id)
            logger.debug(f"Added state {state_id} to active states")
            
            # Update state probability and visit count
            if self.state_service:
                state = self.state_service.get_state(state_id)
                if state:
                    state.set_probability_to_base_probability()
                    state.add_visit()
    
    def remove_active_state(self, state_id: int) -> None:
        """Remove a state from the active state list.
        
        Args:
            state_id: ID of state to deactivate
        """
        if state_id in self.active_states:
            self.active_states.discard(state_id)
            logger.debug(f"Removed state {state_id} from active states")
            
            # Reset state probability
            if self.state_service:
                state = self.state_service.get_state(state_id)
                if state:
                    state.probability_exists = 0
    
    def is_active(self, state_id: int) -> bool:
        """Check if a state is currently active.
        
        Args:
            state_id: State ID to check
            
        Returns:
            True if state is active
        """
        return state_id in self.active_states
    
    def clear_active_states(self) -> None:
        """Clear all active states.
        
        Resets state memory to empty, typically used when starting fresh
        or when the current state is completely unknown.
        """
        for state_id in list(self.active_states):
            self.remove_active_state(state_id)
        logger.info("Cleared all active states")
    
    def set_active_states(self, state_ids: Set[int]) -> None:
        """Replace active states with a new set.
        
        Args:
            state_ids: New set of active state IDs
        """
        self.clear_active_states()
        for state_id in state_ids:
            self.add_active_state(state_id)
    
    def get_active_state_count(self) -> int:
        """Get the number of currently active states.
        
        Returns:
            Count of active states
        """
        return len(self.active_states)
    
    def __str__(self) -> str:
        """String representation."""
        return f"StateMemory(active={self.get_active_state_names_as_string()})"


class StateService:
    """Placeholder for StateService.
    
    Will be implemented when migrating the navigation package.
    """
    
    def get_state(self, state_id: int) -> Optional[State]:
        """Get state by ID.
        
        Args:
            state_id: State ID
            
        Returns:
            State or None
        """
        # Placeholder implementation
        return None