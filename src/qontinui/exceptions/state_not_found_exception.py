"""State not found exception - ported from Qontinui framework.

Exception for missing states.
"""

from .qontinui_runtime_exception import QontinuiRuntimeException


class StateNotFoundException(QontinuiRuntimeException):
    """Thrown when a requested state cannot be found in the state management system.
    
    Port of StateNotFoundException from Qontinui framework class.
    
    This exception indicates that the framework attempted to access or transition
    to a state that doesn't exist in the current state model. This is a critical
    error in model-based automation as it suggests either a configuration problem
    or that the application is in an unexpected state.
    """
    
    def __init__(self, state_name: str, context: str = None):
        """Construct a new state not found exception.
        
        Args:
            state_name: The name of the state that could not be found
            context: Optional additional context about where the state was expected
        """
        self.state_name = state_name
        if context:
            message = f"State '{state_name}' not found in {context}"
        else:
            message = f"State '{state_name}' not found in the state model"
        super().__init__(message)
    
    def get_state_name(self) -> str:
        """Get the name of the state that could not be found.
        
        Returns:
            The state name
        """
        return self.state_name