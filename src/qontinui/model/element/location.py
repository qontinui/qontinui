"""Location - ported from Qontinui framework.

Represents a point on the screen in the framework.
"""

from typing import Optional, Tuple
from dataclasses import dataclass, field
from .position import Position, PositionName


@dataclass
class Location:
    """Represents a point on the screen.
    
    Port of Location from Qontinui framework class.
    
    A Location defines a specific point that can be specified in two ways:
    - Absolute coordinates: Using x,y pixel values directly
    - Relative position: As a percentage position within a Region
    
    Both methods support optional x,y offsets for fine-tuning the final position.
    
    In the model-based approach, Locations are essential for:
    - Specifying click targets within GUI elements
    - Defining anchor points for spatial relationships between elements
    - Positioning the mouse for hover actions
    - Creating dynamic positions that adapt to different screen sizes
    
    The relative positioning feature is particularly powerful in model-based automation
    as it allows locations to remain valid even when GUI elements move or resize, making
    automation more robust to UI changes.
    """
    
    x: int = 0
    """X coordinate in pixels."""
    
    y: int = 0
    """Y coordinate in pixels."""
    
    name: Optional[str] = None
    """Optional name for this location."""
    
    region: Optional['Region'] = None
    """Optional region for relative positioning."""
    
    position: Optional['Position'] = None
    """Position within region (as percentages)."""
    
    anchor: Optional[str] = None
    """Anchor point name."""
    
    offset_x: int = 0
    """X offset in pixels."""
    
    offset_y: int = 0
    """Y offset in pixels."""
    
    def __post_init__(self):
        """Validate location after initialization."""
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
    
    @classmethod
    def from_tuple(cls, coords: Tuple[int, int]) -> 'Location':
        """Create Location from tuple.
        
        Args:
            coords: (x, y) tuple
            
        Returns:
            New Location instance
        """
        return cls(x=coords[0], y=coords[1])
    
    def to_tuple(self) -> Tuple[int, int]:
        """Convert to tuple.
        
        Returns:
            (x, y) tuple
        """
        return (self.x, self.y)
    
    def get_final_location(self) -> 'Location':
        """Get final location after applying region and offsets.
        
        Returns:
            Final computed location
        """
        if self.region and self.position:
            # Calculate position within region
            base_x = self.region.x + int(self.region.width * self.position.percent_w)
            base_y = self.region.y + int(self.region.height * self.position.percent_h)
            return Location(
                x=base_x + self.offset_x,
                y=base_y + self.offset_y,
                name=self.name
            )
        else:
            # Use absolute coordinates with offsets
            return Location(
                x=self.x + self.offset_x,
                y=self.y + self.offset_y,
                name=self.name
            )
    
    def is_defined_with_region(self) -> bool:
        """Check if this location is defined relative to a region.
        
        Returns:
            True if defined with region
        """
        return self.region is not None
    
    def distance_to(self, other: 'Location') -> float:
        """Calculate distance to another location.
        
        Args:
            other: Other location
            
        Returns:
            Euclidean distance
        """
        import math
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)
    
    def offset(self, dx: int, dy: int) -> 'Location':
        """Create new location with offset.
        
        Args:
            dx: X offset
            dy: Y offset
            
        Returns:
            New offset location
        """
        return Location(
            x=self.x + dx,
            y=self.y + dy,
            name=self.name,
            region=self.region,
            position=self.position,
            anchor=self.anchor,
            offset_x=self.offset_x,
            offset_y=self.offset_y
        )
    
    def __str__(self) -> str:
        """String representation."""
        if self.name:
            return f"Location({self.name} at {self.x},{self.y})"
        return f"Location({self.x},{self.y})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Location(x={self.x}, y={self.y}, name={self.name})"