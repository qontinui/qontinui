"""State ID resolver - ported from Qontinui framework.

Service for resolving state names to IDs in state transitions.
"""

from typing import List, Optional
import logging


logger = logging.getLogger(__name__)


class StateIdResolver:
    """Service for resolving state names to IDs in state transitions.
    
    Port of StateIdResolver from Qontinui framework class.
    
    StateIdResolver handles the critical task of converting state names (used in code 
    for readability) to state IDs (used internally for efficient processing). This service 
    specifically processes StateTransitions objects to replace human-readable state names 
    with their corresponding numeric identifiers.
    
    Key responsibilities:
    - Convert state names to IDs within StateTransitions objects
    - Process JavaStateTransition activate lists to resolve state names
    - Perform batch conversion during application initialization
    
    This service is essential for the model-based approach as it bridges the gap between 
    human-readable state definitions and machine-efficient state processing, ensuring both 
    developer productivity and runtime performance.
    """
    
    def __init__(self, all_states_in_project_service: Optional['StateService'] = None):
        """Initialize StateIdResolver.
        
        Args:
            all_states_in_project_service: Service for accessing state definitions
        """
        self.all_states_in_project_service = all_states_in_project_service
    
    def convert_all_state_transitions(self, all_state_transitions: List['StateTransitions']) -> None:
        """Batch convert state names to IDs for all state transitions.
        
        Processes a collection of StateTransitions objects, converting human-readable
        state names to numeric IDs for efficient runtime processing. This bulk
        operation is typically performed during application initialization after
        all states have been registered.
        
        Side effects:
        - Modifies StateTransitions objects in-place
        - Sets state IDs based on registered state names
        - Updates JavaStateTransition activate lists with IDs
        
        Args:
            all_state_transitions: List of StateTransitions to process
        """
        for state_transitions in all_state_transitions:
            self.convert_names_to_ids(state_transitions)
    
    def convert_names_to_ids(self, state_transitions: 'StateTransitions') -> None:
        """Convert state names to IDs within a single StateTransitions object.
        
        Performs name-to-ID conversion for both the parent state and all target
        states referenced in its transitions. This enables the framework to use
        efficient numeric comparisons during runtime while allowing developers
        to define states using meaningful names.
        
        Conversion process:
        1. Convert the parent state name to ID if not already set
        2. For JavaStateTransition objects, convert activate state names
        3. Build parallel lists of state IDs for runtime use
        
        Prerequisites:
        - State names must be unique within the project
        - All referenced states must be registered in StateService
        - StateService must be initialized with all states
        
        Side effects:
        - Sets state_id field if currently None
        - Populates activate ID lists in JavaStateTransition objects
        - Silently ignores unregistered state names
        
        Args:
            state_transitions: The StateTransitions object to process
        """
        if not self.all_states_in_project_service:
            return
        
        # Convert the parent state name to ID
        state_id = self.all_states_in_project_service.get_state_id(state_transitions.state_name)
        if state_transitions.state_id is None and state_id is not None:
            state_transitions.state_id = state_id
        
        # Process each transition
        for transition in state_transitions.get_transitions():
            # Check if this is a JavaStateTransition (has activate_names)
            if hasattr(transition, 'activate_names') and hasattr(transition, 'activate'):
                for state_to_activate in transition.activate_names:
                    state_to_activate_id = self.all_states_in_project_service.get_state_id(
                        state_to_activate
                    )
                    if state_to_activate_id is not None:
                        transition.activate.append(state_to_activate_id)


class StateService:
    """Placeholder for StateService.
    
    Will be implemented when migrating the navigation package.
    """
    
    def get_state_id(self, state_name: str) -> Optional[int]:
        """Get state ID by name.
        
        Args:
            state_name: State name
            
        Returns:
            State ID or None
        """
        return None


class StateTransitions:
    """Placeholder for StateTransitions.
    
    Will be implemented when migrating the navigation package.
    """
    
    def __init__(self):
        """Initialize StateTransitions."""
        self.state_name = ""
        self.state_id = None
        self.transitions = []
    
    def get_transitions(self) -> list:
        """Get transitions.
        
        Returns:
            List of transitions
        """
        return self.transitions