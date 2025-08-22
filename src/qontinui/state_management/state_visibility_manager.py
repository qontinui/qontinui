"""State visibility manager - ported from Qontinui framework.

Manages the conversion of active states to hidden states during transitions.
"""

from typing import Optional
import logging

from .state_memory import StateMemory


logger = logging.getLogger(__name__)


class StateVisibilityManager:
    """Manages the conversion of active states to hidden states.
    
    Port of StateVisibilityManager from Qontinui framework class.
    
    StateVisibilityManager implements a crucial mechanism in the state management system 
    that handles overlapping or layered GUI elements. When a new state becomes active, 
    it may partially or completely obscure previously active states. This class identifies 
    which active states should be "hidden" by the new state and updates their status 
    accordingly.
    
    Key concepts:
    - Hidden States: States that are obscured but still conceptually present
    - Can Hide Relationship: Each state defines which other states it can hide
    - State Layering: Supports GUI elements that overlay each other
    - Back Navigation: Hidden states become targets for "back" operations
    
    The hiding process:
    1. New state becomes active after successful transition
    2. Check new state's canHide list against currently active states
    3. Move matching states from active to hidden status
    4. Update the new state's hidden state list
    5. Remove hidden states from StateMemory's active list
    
    Common scenarios:
    - Modal Dialogs: Dialog hides underlying page but page remains in memory
    - Dropdown Menus: Menu hides portion of page while open
    - Navigation Drawers: Drawer slides over main content
    - Tab Switching: New tab hides previous tab content
    - Popups/Tooltips: Temporary overlays that hide underlying elements
    
    Benefits of hidden state tracking:
    - Enables accurate "back" navigation to previous states
    - Maintains context about GUI layering and hierarchy
    - Supports complex navigation patterns with overlapping elements
    - Prevents false positive state detections from hidden elements
    - Provides foundation for state stack management
    
    Example flow:
        # Initial: MainPage is active
        # User opens settings dialog
        # SettingsDialog.can_hide = [MainPage]
        set_hidden_states.set(SettingsDialog)
        # Result: SettingsDialog active, MainPage hidden
        # User clicks "back"
        # System knows to return to MainPage
    
    In the model-based approach, StateVisibilityManager enables sophisticated state management 
    that mirrors the actual GUI behavior. By tracking which states are hidden rather than 
    inactive, the framework maintains a more accurate model of the GUI's current configuration 
    and can make better navigation decisions.
    
    This hidden state mechanism is essential for:
    - Applications with complex layered interfaces
    - Supporting natural back navigation patterns
    - Maintaining state context during overlay interactions
    - Accurate state detection in multi-layered GUIs
    """
    
    def __init__(self,
                 all_states_in_project_service: Optional['StateService'] = None,
                 state_memory: Optional[StateMemory] = None):
        """Initialize StateVisibilityManager.
        
        Args:
            all_states_in_project_service: Service for accessing state definitions
            state_memory: Current state memory
        """
        self.all_states_in_project_service = all_states_in_project_service
        self.state_memory = state_memory
    
    def set(self, state_to_set: int) -> bool:
        """Process state hiding relationships after a state becomes active.
        
        Examines all currently active states and determines which ones should be
        hidden by the newly activated state based on its canHide configuration.
        States that can be hidden are moved from the active state list to the
        new state's hidden state list, maintaining the layering relationship.
        
        Side effects:
        - Modifies the state_to_set's hidden state list
        - Removes hidden states from StateMemory's active list
        - Preserves hidden states for potential back navigation
        
        Implementation note: Uses list copy to avoid concurrent modification
        while iterating through active states that may be removed.
        
        Args:
            state_to_set: ID of the newly activated state that may hide others
            
        Returns:
            True if state_to_set is valid, False if state not found
        """
        if not self.all_states_in_project_service or not self.state_memory:
            return False
        
        state_opt = self.all_states_in_project_service.get_state(state_to_set)
        if not state_opt:
            return False
        
        state = state_opt
        
        # Use a copy to avoid concurrent modification
        active_states = list(self.state_memory.active_states)
        
        for active_state in active_states:
            if active_state in state.can_hide_ids:
                state.add_hidden_state(active_state)
                self.state_memory.remove_active_state(active_state)
        
        return True


class StateService:
    """Placeholder for StateService.
    
    Will be implemented when migrating the navigation package.
    """
    
    def get_state(self, state_id: int) -> Optional['State']:
        """Get state by ID.
        
        Args:
            state_id: State ID
            
        Returns:
            State or None
        """
        return None