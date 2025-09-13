"""Region offset utilities for declarative positioning.

Provides utilities for calculating and applying offsets to regions.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from ..model.element import Region


@dataclass
class RegionOffset:
    """Represents an offset to apply to a region.
    
    Following Brobot principles:
    - Simple offset representation
    - Can be positive or negative
    - Supports both position and size offsets
    """
    
    x: int = 0
    """X offset (positive = right, negative = left)."""
    
    y: int = 0
    """Y offset (positive = down, negative = up)."""
    
    width: int = 0
    """Width offset (positive = wider, negative = narrower)."""
    
    height: int = 0
    """Height offset (positive = taller, negative = shorter)."""
    
    def apply_to(self, region: Region) -> Region:
        """Apply offset to a region.
        
        Args:
            region: Region to apply offset to
            
        Returns:
            New region with offset applied
        """
        return Region(
            x=region.x + self.x,
            y=region.y + self.y,
            width=max(1, region.width + self.width),
            height=max(1, region.height + self.height),
            name=region.name
        )
    
    def __add__(self, other: 'RegionOffset') -> 'RegionOffset':
        """Add two offsets together.
        
        Args:
            other: Another offset
            
        Returns:
            Combined offset
        """
        return RegionOffset(
            x=self.x + other.x,
            y=self.y + other.y,
            width=self.width + other.width,
            height=self.height + other.height
        )
    
    def __neg__(self) -> 'RegionOffset':
        """Negate the offset.
        
        Returns:
            Negated offset
        """
        return RegionOffset(
            x=-self.x,
            y=-self.y,
            width=-self.width,
            height=-self.height
        )
    
    def __mul__(self, factor: float) -> 'RegionOffset':
        """Scale the offset.
        
        Args:
            factor: Scaling factor
            
        Returns:
            Scaled offset
        """
        return RegionOffset(
            x=int(self.x * factor),
            y=int(self.y * factor),
            width=int(self.width * factor),
            height=int(self.height * factor)
        )
    
    @classmethod
    def position(cls, x: int, y: int) -> 'RegionOffset':
        """Create a position-only offset.
        
        Args:
            x: X offset
            y: Y offset
            
        Returns:
            Position offset
        """
        return cls(x=x, y=y, width=0, height=0)
    
    @classmethod
    def size(cls, width: int, height: int) -> 'RegionOffset':
        """Create a size-only offset.
        
        Args:
            width: Width offset
            height: Height offset
            
        Returns:
            Size offset
        """
        return cls(x=0, y=0, width=width, height=height)
    
    @classmethod
    def uniform(cls, amount: int) -> 'RegionOffset':
        """Create a uniform offset in all dimensions.
        
        Args:
            amount: Amount to offset
            
        Returns:
            Uniform offset
        """
        return cls(x=amount, y=amount, width=amount, height=amount)