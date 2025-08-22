"""State registration service - ported from Qontinui framework.

Handles registration of states with the state management system.
"""

from typing import Set
import logging
from ..state_management.state import State
from ..state_management.state_service import StateService

logger = logging.getLogger(__name__)


class StateRegistrationService:
    """Service for registering states discovered through annotations.
    
    Port of StateRegistrationService from Qontinui framework.
    
    This service handles the registration of states with the StateService,
    ensuring that:
    - States are properly validated before registration
    - Duplicate states are not registered
    - Registration failures are properly logged
    - State counts are accurately tracked
    """
    
    def __init__(self, state_service: StateService):
        """Initialize the registration service.
        
        Args:
            state_service: Service for state management
        """
        self.state_service = state_service
        self._registered_states: Set[str] = set()
    
    def register_state(self, state: State) -> bool:
        """Register a state with the state service.
        
        Args:
            state: State to register
            
        Returns:
            True if registration successful, False otherwise
        """
        if not state:
            logger.error("Cannot register null state")
            return False
        
        state_name = state.name
        
        # Check for duplicate registration
        if state_name in self._registered_states:
            logger.warning(f"State '{state_name}' is already registered")
            return False
        
        # Validate state
        if not self._validate_state(state):
            logger.error(f"State '{state_name}' failed validation")
            return False
        
        try:
            # Register with state service
            self.state_service.add_state(state)
            self._registered_states.add(state_name)
            
            logger.info(f"Successfully registered state: {state_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register state '{state_name}'", exc_info=e)
            return False
    
    def _validate_state(self, state: State) -> bool:
        """Validate a state before registration.
        
        Args:
            state: State to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Basic validation
        if not state.name:
            logger.error("State has no name")
            return False
        
        # Check for at least one component
        has_components = False
        
        if hasattr(state, 'state_images') and state.state_images:
            has_components = True
        elif hasattr(state, 'state_strings') and state.state_strings:
            has_components = True
        elif hasattr(state, 'state_objects') and state.state_objects:
            has_components = True
        
        if not has_components:
            logger.warning(f"State '{state.name}' has no components")
            # This is allowed but worth warning about
        
        return True
    
    def get_registered_state_count(self) -> int:
        """Get the number of registered states.
        
        Returns:
            Number of successfully registered states
        """
        return len(self._registered_states)
    
    def is_registered(self, state_name: str) -> bool:
        """Check if a state is registered.
        
        Args:
            state_name: Name of the state
            
        Returns:
            True if registered, False otherwise
        """
        return state_name in self._registered_states
    
    def clear_registrations(self) -> None:
        """Clear all registration records.
        
        This doesn't remove states from StateService,
        just clears the registration tracking.
        """
        self._registered_states.clear()
        logger.info("Cleared all state registrations")