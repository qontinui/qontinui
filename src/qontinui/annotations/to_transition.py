"""ToTransition annotation for Qontinui framework.

Marks methods as ToTransitions (also known as arrival transitions or finish transitions).
This transition verifies that we have successfully arrived at the target state.
Ported from Brobot's ToTransition annotation.
"""

from typing import Any, Optional
from functools import wraps


def to_transition(
    description: str = "",
    timeout: int = 5,
    required: bool = True
) -> Any:
    """Marks a method as a ToTransition.
    
    A ToTransition (also known as arrival transition or finish transition)
    verifies that we have successfully arrived at the target state.
    
    The annotated method should:
    - Return boolean (true if state is confirmed, false otherwise)
    - Verify the presence of unique elements that confirm the state is active
    - Be a member of a class annotated with @transition_set
    - There should be only ONE @to_transition method per @transition_set class
    
    This transition is executed after any FromTransition to confirm successful
    navigation to the target state, regardless of which state we came from.
    
    Example usage:
        @to_transition
        def verify_arrival(self) -> bool:
            logger.info("Verifying arrival at Pricing state")
            return self.action.find(self.pricing_state.start_for_free_button).is_success()
    
    Args:
        description: Optional description of this arrival verification.
                    Useful for documentation and debugging.
        timeout: Timeout for verifying arrival in seconds. Default is 5 seconds.
        required: Whether this verification is required for the transition
                 to be considered successful. If false, failure of this
                 verification will log a warning but not fail the transition.
                 Default is True.
    
    Returns:
        The decorated method with transition metadata attached.
    
    Since: 1.2.0
    """
    def decorator(method: callable) -> callable:
        # Store metadata on the method
        method._qontinui_to_transition = True
        method._qontinui_to_transition_description = description
        method._qontinui_to_transition_timeout = timeout
        method._qontinui_to_transition_required = required
        
        @wraps(method)
        def wrapper(*args, **kwargs):
            return method(*args, **kwargs)
        
        # Copy metadata to wrapper
        wrapper._qontinui_to_transition = True
        wrapper._qontinui_to_transition_description = description
        wrapper._qontinui_to_transition_timeout = timeout
        wrapper._qontinui_to_transition_required = required
        
        return wrapper
    
    return decorator


def is_to_transition(method: Any) -> bool:
    """Check if a method is a ToTransition.
    
    Args:
        method: Method to check
        
    Returns:
        True if method is decorated with @to_transition
    """
    return hasattr(method, '_qontinui_to_transition') and method._qontinui_to_transition


def get_to_transition_metadata(method: Any) -> Optional[dict]:
    """Get ToTransition metadata from a decorated method.
    
    Args:
        method: The transition method
        
    Returns:
        Dictionary with transition metadata or None
    """
    if not is_to_transition(method):
        return None
    
    return {
        'description': getattr(method, '_qontinui_to_transition_description', ''),
        'timeout': getattr(method, '_qontinui_to_transition_timeout', 5),
        'required': getattr(method, '_qontinui_to_transition_required', True)
    }