"""TransitionSet annotation for Qontinui framework.

Marks classes as containing all transitions for a specific state.
Ported from Brobot's TransitionSet annotation.
"""

from typing import Type, Any, Optional
from functools import wraps


def transition_set(
    state: Type,
    name: str = "",
    description: str = ""
) -> Any:
    """Marks a class as containing all transitions for a specific state.
    
    This annotation groups all FromTransitions and the ToTransition for a state
    in one class, maintaining high cohesion and making it easy to understand
    all paths to and from a state.
    
    Classes annotated with @transition_set should contain:
    - Methods annotated with @from_transition for transitions FROM other states TO this state
    - One method annotated with @to_transition to verify arrival at this state
    
    Example usage:
        @transition_set(state=PricingState)
        class PricingTransitions:
            def __init__(self, menu_state, pricing_state, action):
                self.menu_state = menu_state
                self.pricing_state = pricing_state
                self.action = action
            
            @from_transition(from_state=MenuState)
            def from_menu(self) -> bool:
                return self.action.click(self.menu_state.pricing_button).is_success()
            
            @from_transition(from_state=HomepageState)
            def from_homepage(self) -> bool:
                return self.action.click(self.homepage_state.pricing_link).is_success()
            
            @to_transition
            def verify_arrival(self) -> bool:
                return self.action.find(self.pricing_state.unique_element).is_success()
    
    Args:
        state: The state class that these transitions belong to.
               All transitions in this class will navigate TO this state.
        name: Optional name override for the state. If not specified,
              the state name will be derived from the state class name.
        description: Optional description of this transition set.
                    Useful for documentation and debugging.
    
    Returns:
        The decorated class with transition set metadata attached.
    
    Since: 1.2.0
    """
    def decorator(cls: Type) -> Type:
        # Store metadata on the class
        cls._qontinui_transition_set = True
        cls._qontinui_transition_set_state = state
        cls._qontinui_transition_set_name = name
        cls._qontinui_transition_set_description = description
        
        return cls
    
    return decorator


def is_transition_set(obj: Any) -> bool:
    """Check if an object is a Qontinui transition set.
    
    Args:
        obj: Object to check
        
    Returns:
        True if object is decorated with @transition_set
    """
    return hasattr(obj, '_qontinui_transition_set') and obj._qontinui_transition_set


def get_transition_set_metadata(cls: Type) -> Optional[dict]:
    """Get transition set metadata from a decorated class.
    
    Args:
        cls: The transition set class
        
    Returns:
        Dictionary with transition set metadata or None
    """
    if not is_transition_set(cls):
        return None
    
    return {
        'state': getattr(cls, '_qontinui_transition_set_state', None),
        'name': getattr(cls, '_qontinui_transition_set_name', ''),
        'description': getattr(cls, '_qontinui_transition_set_description', '')
    }