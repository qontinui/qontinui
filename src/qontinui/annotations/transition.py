"""Transition annotation for Qontinui framework.

Marks classes as Qontinui transitions for automatic registration.
"""

from typing import List, Type, Optional, Any, Union
from functools import wraps


def transition(
    from_states: Union[Type, List[Type]],
    to_states: Union[Type, List[Type]],
    method: str = "execute",
    description: str = "",
    priority: int = 0
) -> Any:
    """Annotation for Qontinui transitions.
    
    This decorator marks a class as a Qontinui transition and enables
    automatic registration with the state transition system.
    
    Classes decorated with @transition should include:
    - An execute method (or custom method name) that performs the transition
    - Dependencies injected through constructor
    
    Usage:
        @transition(from_states=PromptState, to_states=WorkingState)
        class PromptToWorkingTransition:
            def __init__(self, action_config):
                self.action_config = action_config
            
            def execute(self) -> bool:
                # transition logic
                return True
    
    For transitions with multiple targets:
        @transition(
            from_states=WorkingState,
            to_states=[ResultState, ErrorState]
        )
        class WorkingTransitions:
            # transition logic
    
    Args:
        from_states: The source state class(es) for this transition.
                    Can be a single class or list of classes.
        to_states: The target state class(es) for this transition.
                  Can be a single class or list of classes.
        method: The method name that executes the transition logic.
               Defaults to "execute". The method should return bool
               or StateTransition.
        description: Optional description of the transition's purpose.
                    Used for documentation and debugging.
        priority: Priority of this transition when multiple transitions
                 are possible. Higher values indicate higher priority.
    
    Returns:
        The decorated class with transition metadata attached.
    """
    def decorator(cls: Type) -> Type:
        # Normalize to lists
        from_list = from_states if isinstance(from_states, list) else [from_states]
        to_list = to_states if isinstance(to_states, list) else [to_states]
        
        # Store metadata on the class
        cls._qontinui_transition = True
        cls._qontinui_transition_from = from_list
        cls._qontinui_transition_to = to_list
        cls._qontinui_transition_method = method
        cls._qontinui_transition_description = description
        cls._qontinui_transition_priority = priority
        
        # Validate that the method exists
        if not hasattr(cls, method):
            # Create a placeholder that will be checked at runtime
            setattr(cls, '_qontinui_transition_method_missing', True)
        
        return cls
    
    return decorator


def is_transition(obj: Any) -> bool:
    """Check if an object is a Qontinui transition.
    
    Args:
        obj: Object to check
        
    Returns:
        True if object is decorated with @transition
    """
    return hasattr(obj, '_qontinui_transition') and obj._qontinui_transition


def get_transition_metadata(cls: Type) -> Optional[dict]:
    """Get transition metadata from a decorated class.
    
    Args:
        cls: The transition class
        
    Returns:
        Dictionary with transition metadata or None
    """
    if not is_transition(cls):
        return None
    
    return {
        'from_states': getattr(cls, '_qontinui_transition_from', []),
        'to_states': getattr(cls, '_qontinui_transition_to', []),
        'method': getattr(cls, '_qontinui_transition_method', 'execute'),
        'description': getattr(cls, '_qontinui_transition_description', ''),
        'priority': getattr(cls, '_qontinui_transition_priority', 0)
    }