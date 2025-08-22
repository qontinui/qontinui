"""Movement class - ported from Qontinui framework.

Represents a directed movement from a start location to an end location.
"""

from dataclasses import dataclass
from typing import Optional
import math
from .location import Location


@dataclass(frozen=True)
class Movement:
    """Represents a directed movement from a start location to an end location.
    
    Port of Movement from Qontinui framework class.
    
    This class is essential for representing actions that have a clear direction and
    path, such as a mouse drag or a screen swipe. Unlike a Region, which only defines
    a static area, a Movement encapsulates the dynamic concept of a transition
    between two points.
    
    It is an immutable object, ensuring that the definition of a movement cannot be
    changed after it is created.
    """
    
    start_location: Location
    end_location: Location
    
    def __post_init__(self):
        """Validate locations are not None."""
        if self.start_location is None:
            raise ValueError("Start location cannot be null")
        if self.end_location is None:
            raise ValueError("End location cannot be null")
    
    @property
    def delta_x(self) -> int:
        """Calculate the horizontal displacement of the movement.
        
        Returns:
            The change in the x-coordinate (end - start)
        """
        return self.end_location.x - self.start_location.x
    
    @property
    def delta_y(self) -> int:
        """Calculate the vertical displacement of the movement.
        
        Returns:
            The change in the y-coordinate (end - start)
        """
        return self.end_location.y - self.start_location.y
    
    @property
    def distance(self) -> float:
        """Calculate the Euclidean distance of the movement.
        
        Returns:
            The straight-line distance between start and end
        """
        return math.sqrt(self.delta_x ** 2 + self.delta_y ** 2)
    
    @property
    def angle(self) -> float:
        """Calculate the angle of the movement in radians.
        
        Returns:
            Angle in radians from horizontal (0 is right, pi/2 is up)
        """
        return math.atan2(-self.delta_y, self.delta_x)  # Negative because Y increases downward
    
    @property
    def angle_degrees(self) -> float:
        """Calculate the angle of the movement in degrees.
        
        Returns:
            Angle in degrees from horizontal (0 is right, 90 is up)
        """
        return math.degrees(self.angle)
    
    def reverse(self) -> 'Movement':
        """Create a reversed movement (end becomes start, start becomes end).
        
        Returns:
            New Movement with reversed direction
        """
        return Movement(self.end_location, self.start_location)
    
    def scale(self, factor: float) -> 'Movement':
        """Create a scaled movement maintaining the same start point.
        
        Args:
            factor: Scale factor (1.0 = same size, 2.0 = double, 0.5 = half)
            
        Returns:
            New Movement with scaled distance
        """
        new_x = int(self.start_location.x + self.delta_x * factor)
        new_y = int(self.start_location.y + self.delta_y * factor)
        return Movement(
            self.start_location,
            Location(new_x, new_y)
        )
    
    def translate(self, dx: int, dy: int) -> 'Movement':
        """Create a translated movement (both points moved by same amount).
        
        Args:
            dx: Horizontal translation
            dy: Vertical translation
            
        Returns:
            New Movement translated by (dx, dy)
        """
        return Movement(
            Location(self.start_location.x + dx, self.start_location.y + dy),
            Location(self.end_location.x + dx, self.end_location.y + dy)
        )
    
    def is_horizontal(self, tolerance: int = 5) -> bool:
        """Check if movement is primarily horizontal.
        
        Args:
            tolerance: Maximum Y change to consider horizontal
            
        Returns:
            True if movement is mostly horizontal
        """
        return abs(self.delta_y) <= tolerance
    
    def is_vertical(self, tolerance: int = 5) -> bool:
        """Check if movement is primarily vertical.
        
        Args:
            tolerance: Maximum X change to consider vertical
            
        Returns:
            True if movement is mostly vertical
        """
        return abs(self.delta_x) <= tolerance
    
    def is_diagonal(self, tolerance: float = 0.1) -> bool:
        """Check if movement is diagonal (45 degrees).
        
        Args:
            tolerance: Ratio tolerance for considering diagonal
            
        Returns:
            True if movement is roughly diagonal
        """
        if self.delta_x == 0:
            return False
        ratio = abs(self.delta_y / self.delta_x)
        return abs(ratio - 1.0) <= tolerance
    
    def __str__(self) -> str:
        """String representation."""
        return f"Movement[from={self.start_location}, to={self.end_location}]"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"Movement(start_location={self.start_location!r}, end_location={self.end_location!r})"
    
    @classmethod
    def from_coordinates(cls, x1: int, y1: int, x2: int, y2: int) -> 'Movement':
        """Create Movement from coordinate pairs.
        
        Args:
            x1: Start X coordinate
            y1: Start Y coordinate
            x2: End X coordinate
            y2: End Y coordinate
            
        Returns:
            New Movement instance
        """
        return cls(Location(x1, y1), Location(x2, y2))
    
    @classmethod
    def from_delta(cls, start: Location, dx: int, dy: int) -> 'Movement':
        """Create Movement from start point and deltas.
        
        Args:
            start: Starting location
            dx: Horizontal displacement
            dy: Vertical displacement
            
        Returns:
            New Movement instance
        """
        return cls(start, Location(start.x + dx, start.y + dy))