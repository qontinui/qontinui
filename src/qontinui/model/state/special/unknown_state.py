"""UnknownState class - ported from Qontinui framework.

Represents the initial uncertain state in the state management system.
"""

from typing import Optional
from enum import Enum
from ..state import State
from ..state_enum import StateEnum


class UnknownStateEnum(Enum):
    """Enum for UnknownState.
    
    Port of UnknownState from Qontinui framework.Enum.
    Implements StateEnum interface.
    """
    UNKNOWN = "UNKNOWN"
    
    @property
    def name(self) -> str:
        """Get state name."""
        return self.value


class UnknownState:
    """Represents the initial uncertain state in the state management system.
    
    Port of UnknownState from Qontinui framework class.
    
    UnknownState serves as the universal entry point and recovery mechanism in the 
    model-based automation framework. When the system starts or loses track of the current 
    application state, it begins in the Unknown state. This state acts as a safety net, 
    ensuring the automation can always recover from unexpected situations and navigate 
    back to known states.
    
    Key characteristics:
    - Universal Entry: Default starting state for all automations
    - Recovery Point: Fallback when state detection fails
    - No Prerequisites: Always accessible regardless of application state
    - Transition Hub: Should have paths to major application states
    - Error Resilient: Handles unexpected application conditions
    
    Common scenarios leading to Unknown state:
    - Initial automation startup
    - Application crash or restart
    - Unexpected popups or dialogs
    - Network errors disrupting navigation
    - State detection confidence below threshold
    - Manual intervention during automation
    
    Best practices for Unknown state transitions:
    - Include paths to main menu or home screens
    - Add error dismissal actions (close dialogs, alerts)
    - Implement application restart procedures
    - Use robust visual patterns that work across contexts
    - Consider multiple recovery strategies
    
    Example recovery strategies:
    - ESC key to close potential dialogs
    - Alt+F4 to close unknown windows
    - Click on application icon to ensure focus
    - Navigate to known URL or home screen
    - Use keyboard shortcuts to reach main menu
    """
    
    _instance: Optional['UnknownState'] = None
    
    def __init__(self, state_service: Optional['StateService'] = None):
        """Initialize UnknownState.
        
        Args:
            state_service: Optional StateService to register with
        """
        self._state = State(name="unknown", state_enum=UnknownStateEnum.UNKNOWN)
        if state_service:
            state_service.save(self._state)
    
    @property
    def state(self) -> State:
        """Get the unknown state.
        
        Returns:
            The unknown State instance
        """
        return self._state
    
    @property
    def enum(self) -> UnknownStateEnum:
        """Get the state enum.
        
        Returns:
            UnknownStateEnum.UNKNOWN
        """
        return UnknownStateEnum.UNKNOWN
    
    def is_unknown(self) -> bool:
        """Check if this is an unknown state.
        
        Returns:
            Always True for UnknownState
        """
        return True
    
    def add_recovery_transition(self, target_state: State, action: 'Action') -> None:
        """Add a recovery transition to a known state.
        
        Args:
            target_state: State to transition to
            action: Action to perform for transition
        """
        from ..state_transition import StateTransition
        transition = StateTransition(
            from_state=self._state,
            to_state=target_state,
            action=action
        )
        self._state.add_transition(transition)
    
    def clear_recovery_transitions(self) -> None:
        """Clear all recovery transitions."""
        self._state.transitions.clear()
    
    def __str__(self) -> str:
        """String representation."""
        return "UnknownState"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"UnknownState(state={self._state.name})"
    
    @classmethod
    def instance(cls, state_service: Optional['StateService'] = None) -> 'UnknownState':
        """Get or create the UnknownState singleton.
        
        Args:
            state_service: Optional StateService to register with
            
        Returns:
            The UnknownState singleton instance
        """
        if cls._instance is None:
            cls._instance = cls(state_service)
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None