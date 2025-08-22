"""States registered event - ported from Qontinui framework.

Event published when states have been registered through annotations.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class StatesRegisteredEvent:
    """Event published when states are registered.
    
    Port of StatesRegisteredEvent from Qontinui framework.
    
    This event is published by the AnnotationProcessor after it has
    successfully processed all @state and @transition annotations.
    It allows other components to react to the completion of state
    registration.
    """
    
    source: Any
    """The source that published the event (usually AnnotationProcessor)."""
    
    state_count: int
    """Number of states that were registered."""
    
    transition_count: int
    """Number of transitions that were registered."""
    
    def __str__(self) -> str:
        """String representation of the event.
        
        Returns:
            Human-readable event description
        """
        return (
            f"StatesRegisteredEvent(states={self.state_count}, "
            f"transitions={self.transition_count})"
        )