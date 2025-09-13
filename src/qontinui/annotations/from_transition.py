"""FromTransition annotation for Qontinui framework.

Marks methods as transitions FROM a specific state TO the state defined
in the enclosing @TransitionSet class.
Ported from Brobot's FromTransition annotation.
"""

from typing import Type, Any, Optional
from functools import wraps


def from_transition(
    from_state: Type,
    priority: int = 0,
    description: str = "",
    timeout: int = 10
) -> Any:
    """Marks a method as a FromTransition.
    
    A FromTransition is a transition FROM a specific state TO the state defined
    in the enclosing @transition_set class.
    
    The annotated method should:
    - Return boolean (true if transition succeeds, false otherwise)
    - Contain the actions needed to navigate FROM the source state
    - Be a member of a class annotated with @transition_set
    
    Example usage:
        @from_transition(from_state=MenuState, priority=1)
        def from_menu(self) -> bool:
            logger.info("Navigating from Menu to Pricing")
            return self.action.click(self.menu_state.pricing_button).is_success()
    
    Args:
        from_state: The source state class for this transition.
                   This transition will navigate FROM this state TO
                   the state defined in @transition_set.
        priority: Priority of this transition when multiple paths exist.
                 Higher values indicate higher priority. Default is 0.
        description: Optional description of this transition.
                    Useful for documentation and debugging.
        timeout: Timeout for this transition in seconds. Default is 10 seconds.
    
    Returns:
        The decorated method with transition metadata attached.
    
    Since: 1.2.0
    """
    def decorator(method: callable) -> callable:
        # Store metadata on the method
        method._qontinui_from_transition = True
        method._qontinui_from_transition_from = from_state
        method._qontinui_from_transition_priority = priority
        method._qontinui_from_transition_description = description
        method._qontinui_from_transition_timeout = timeout
        
        @wraps(method)
        def wrapper(*args, **kwargs):
            return method(*args, **kwargs)
        
        # Copy metadata to wrapper
        wrapper._qontinui_from_transition = True
        wrapper._qontinui_from_transition_from = from_state
        wrapper._qontinui_from_transition_priority = priority
        wrapper._qontinui_from_transition_description = description
        wrapper._qontinui_from_transition_timeout = timeout
        
        return wrapper
    
    return decorator


def is_from_transition(method: Any) -> bool:
    """Check if a method is a FromTransition.
    
    Args:
        method: Method to check
        
    Returns:
        True if method is decorated with @from_transition
    """
    return hasattr(method, '_qontinui_from_transition') and method._qontinui_from_transition


def get_from_transition_metadata(method: Any) -> Optional[dict]:
    """Get FromTransition metadata from a decorated method.
    
    Args:
        method: The transition method
        
    Returns:
        Dictionary with transition metadata or None
    """
    if not is_from_transition(method):
        return None
    
    return {
        'from_state': getattr(method, '_qontinui_from_transition_from', None),
        'priority': getattr(method, '_qontinui_from_transition_priority', 0),
        'description': getattr(method, '_qontinui_from_transition_description', ''),
        'timeout': getattr(method, '_qontinui_from_transition_timeout', 10)
    }