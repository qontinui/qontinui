"""StateLocation - ported from Qontinui framework.

Locations associated with states.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from ..element import Location


@dataclass
class StateLocation:
    """Location associated with a state.
    
    Port of StateLocation from Qontinui framework class.
    Represents a specific point in a state.
    """
    
    location: Location
    name: Optional[str] = None
    owner_state: Optional['State'] = None
    
    # Location properties
    _anchor: bool = False  # If true, used as anchor point
    _fixed: bool = True  # If true, location is fixed
    _click_target: bool = False  # If true, used as click target
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize location name."""
        if self.name is None:
            self.name = f"Location_{self.location.x}_{self.location.y}"
    
    def click(self) -> 'ActionResult':
        """Click at this location.
        
        Returns:
            ActionResult from click
        """
        from ..actions import Action, ClickOptions
        action = Action(ClickOptions())
        return action.click(self.location.x, self.location.y)
    
    def hover(self) -> 'ActionResult':
        """Move mouse to this location.
        
        Returns:
            ActionResult from move
        """
        from ..actions import Action
        action = Action()
        return action.move(self.location.x, self.location.y)
    
    def distance_to(self, other: 'StateLocation') -> float:
        """Calculate distance to another location.
        
        Args:
            other: Other StateLocation
            
        Returns:
            Distance in pixels
        """
        return self.location.distance_to(other.location)
    
    def set_anchor(self, anchor: bool = True) -> 'StateLocation':
        """Set whether this is an anchor point (fluent).
        
        Args:
            anchor: True if anchor point
            
        Returns:
            Self for chaining
        """
        self._anchor = anchor
        return self
    
    def set_fixed(self, fixed: bool = True) -> 'StateLocation':
        """Set whether location is fixed (fluent).
        
        Args:
            fixed: True if fixed
            
        Returns:
            Self for chaining
        """
        self._fixed = fixed
        return self
    
    def set_click_target(self, click_target: bool = True) -> 'StateLocation':
        """Set whether this is a click target (fluent).
        
        Args:
            click_target: True if click target
            
        Returns:
            Self for chaining
        """
        self._click_target = click_target
        return self
    
    @property
    def is_anchor(self) -> bool:
        """Check if this is an anchor point."""
        return self._anchor
    
    @property
    def is_fixed(self) -> bool:
        """Check if location is fixed."""
        return self._fixed
    
    @property
    def is_click_target(self) -> bool:
        """Check if this is a click target."""
        return self._click_target
    
    def __str__(self) -> str:
        """String representation."""
        state_name = self.owner_state.name if self.owner_state else "None"
        return f"StateLocation('{self.name}' at {self.location} in state '{state_name}')"