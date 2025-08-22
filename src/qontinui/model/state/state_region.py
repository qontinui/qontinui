"""StateRegion - ported from Qontinui framework.

Regions associated with states.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from ..element import Region


@dataclass
class StateRegion:
    """Region associated with a state.
    
    Port of StateRegion from Qontinui framework class.
    Represents a region that is part of a state.
    """
    
    region: Region
    name: Optional[str] = None
    owner_state: Optional['State'] = None
    
    # Region properties
    _fixed: bool = True  # If true, region is fixed in position
    _search_region: bool = False  # If true, used as search region for state
    _interaction_region: bool = False  # If true, used for interactions
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize region name."""
        if self.name is None:
            self.name = f"Region_{self.region.x}_{self.region.y}"
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is in region.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if point is in region
        """
        from ..element import Location
        return self.region.contains(Location(x, y))
    
    def get_center(self) -> 'Location':
        """Get center of region.
        
        Returns:
            Center location
        """
        return self.region.center
    
    def click(self) -> 'ActionResult':
        """Click in the center of this region.
        
        Returns:
            ActionResult from click
        """
        from ..actions import Action, ClickOptions
        center = self.get_center()
        action = Action(ClickOptions())
        return action.click(center.x, center.y)
    
    def hover(self) -> 'ActionResult':
        """Move mouse to center of region.
        
        Returns:
            ActionResult from move
        """
        from ..actions import Action
        center = self.get_center()
        action = Action()
        return action.move(center.x, center.y)
    
    def set_fixed(self, fixed: bool = True) -> 'StateRegion':
        """Set whether region is fixed (fluent).
        
        Args:
            fixed: True if region is fixed
            
        Returns:
            Self for chaining
        """
        self._fixed = fixed
        return self
    
    def set_search_region(self, search: bool = True) -> 'StateRegion':
        """Set whether this is a search region (fluent).
        
        Args:
            search: True if search region
            
        Returns:
            Self for chaining
        """
        self._search_region = search
        return self
    
    def set_interaction_region(self, interaction: bool = True) -> 'StateRegion':
        """Set whether this is an interaction region (fluent).
        
        Args:
            interaction: True if interaction region
            
        Returns:
            Self for chaining
        """
        self._interaction_region = interaction
        return self
    
    @property
    def is_fixed(self) -> bool:
        """Check if region is fixed."""
        return self._fixed
    
    @property
    def is_search_region(self) -> bool:
        """Check if this is a search region."""
        return self._search_region
    
    @property
    def is_interaction_region(self) -> bool:
        """Check if this is an interaction region."""
        return self._interaction_region
    
    def __str__(self) -> str:
        """String representation."""
        state_name = self.owner_state.name if self.owner_state else "None"
        return f"StateRegion('{self.name}' in state '{state_name}')"