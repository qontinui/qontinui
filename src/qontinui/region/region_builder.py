"""Fluent builder for declarative region definition.

Provides an intuitive API for defining regions with
relative positioning, sizing, and relationships.
"""

from typing import Optional, Union, Callable
from ..model.element import Region
from .anchor import Anchor, AnchorPoint
from .region_offset import RegionOffset


class RegionBuilder:
    """Fluent builder for creating regions declaratively.
    
    Following Brobot principles:
    - Fluent interface for readability
    - Relative positioning to other regions
    - Anchor-based alignment
    - Intuitive offset and sizing
    
    Example usage:
        # Define a search region below a button
        search_region = (RegionBuilder()
            .below(button_region)
            .with_offset(0, 10)
            .with_size(300, 50)
            .build())
        
        # Define region relative to screen
        sidebar = (RegionBuilder()
            .at_screen_anchor(AnchorPoint.SCREEN_TOP_LEFT)
            .with_size(200, screen_height)
            .build())
    """
    
    def __init__(self):
        """Initialize the region builder."""
        self._x: int = 0
        self._y: int = 0
        self._width: int = 100
        self._height: int = 100
        self._name: Optional[str] = None
        
        # Reference region for relative positioning
        self._reference: Optional[Region] = None
        self._reference_anchor: Optional[AnchorPoint] = None
        self._self_anchor: Optional[AnchorPoint] = None
        
        # Offsets
        self._offset_x: int = 0
        self._offset_y: int = 0
        
        # Padding/margins
        self._padding: int = 0
        self._margin: int = 0
    
    def named(self, name: str) -> 'RegionBuilder':
        """Set the region name.
        
        Args:
            name: Name for the region
            
        Returns:
            Self for chaining
        """
        self._name = name
        return self
    
    def at(self, x: int, y: int) -> 'RegionBuilder':
        """Set absolute position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Self for chaining
        """
        self._x = x
        self._y = y
        return self
    
    def with_size(self, width: int, height: int) -> 'RegionBuilder':
        """Set region size.
        
        Args:
            width: Width in pixels
            height: Height in pixels
            
        Returns:
            Self for chaining
        """
        self._width = width
        self._height = height
        return self
    
    def with_dimensions(self, x: int, y: int, width: int, height: int) -> 'RegionBuilder':
        """Set complete dimensions.
        
        Args:
            x: X coordinate
            y: Y coordinate
            width: Width
            height: Height
            
        Returns:
            Self for chaining
        """
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        return self
    
    def from_region(self, region: Region) -> 'RegionBuilder':
        """Copy dimensions from another region.
        
        Args:
            region: Region to copy from
            
        Returns:
            Self for chaining
        """
        self._x = region.x
        self._y = region.y
        self._width = region.width
        self._height = region.height
        return self
    
    # Relative positioning methods
    
    def relative_to(self, region: Region) -> 'RegionBuilder':
        """Set reference region for relative positioning.
        
        Args:
            region: Reference region
            
        Returns:
            Self for chaining
        """
        self._reference = region
        return self
    
    def above(self, region: Region, gap: int = 0) -> 'RegionBuilder':
        """Position above another region.
        
        Args:
            region: Reference region
            gap: Gap between regions in pixels
            
        Returns:
            Self for chaining
        """
        self._reference = region
        self._x = region.x
        self._y = region.y - self._height - gap
        return self
    
    def below(self, region: Region, gap: int = 0) -> 'RegionBuilder':
        """Position below another region.
        
        Args:
            region: Reference region
            gap: Gap between regions in pixels
            
        Returns:
            Self for chaining
        """
        self._reference = region
        self._x = region.x
        self._y = region.y + region.height + gap
        return self
    
    def left_of(self, region: Region, gap: int = 0) -> 'RegionBuilder':
        """Position to the left of another region.
        
        Args:
            region: Reference region
            gap: Gap between regions in pixels
            
        Returns:
            Self for chaining
        """
        self._reference = region
        self._x = region.x - self._width - gap
        self._y = region.y
        return self
    
    def right_of(self, region: Region, gap: int = 0) -> 'RegionBuilder':
        """Position to the right of another region.
        
        Args:
            region: Reference region
            gap: Gap between regions in pixels
            
        Returns:
            Self for chaining
        """
        self._reference = region
        self._x = region.x + region.width + gap
        self._y = region.y
        return self
    
    def inside(self, region: Region) -> 'RegionBuilder':
        """Position inside another region.
        
        Args:
            region: Container region
            
        Returns:
            Self for chaining
        """
        self._reference = region
        self._x = region.x + self._margin
        self._y = region.y + self._margin
        
        # Adjust size to fit
        max_width = region.width - 2 * self._margin
        max_height = region.height - 2 * self._margin
        
        if self._width > max_width:
            self._width = max_width
        if self._height > max_height:
            self._height = max_height
        
        return self
    
    def centered_in(self, region: Region) -> 'RegionBuilder':
        """Center inside another region.
        
        Args:
            region: Container region
            
        Returns:
            Self for chaining
        """
        self._reference = region
        self._x = region.x + (region.width - self._width) // 2
        self._y = region.y + (region.height - self._height) // 2
        return self
    
    # Anchor-based positioning
    
    def align_to_anchor(self, region: Region, 
                       from_anchor: AnchorPoint,
                       to_anchor: AnchorPoint) -> 'RegionBuilder':
        """Align to a region using anchor points.
        
        Args:
            region: Target region
            from_anchor: Anchor on this region
            to_anchor: Anchor on target region
            
        Returns:
            Self for chaining
        """
        self._reference = region
        self._reference_anchor = to_anchor
        self._self_anchor = from_anchor
        
        # Calculate position based on anchors
        temp_region = Region(self._x, self._y, self._width, self._height)
        aligned = Anchor.align_to(temp_region, region, from_anchor, to_anchor)
        
        self._x = aligned.x
        self._y = aligned.y
        
        return self
    
    def at_screen_anchor(self, anchor: AnchorPoint) -> 'RegionBuilder':
        """Position at a screen anchor point.
        
        Args:
            anchor: Screen anchor point
            
        Returns:
            Self for chaining
        """
        x, y = Anchor._get_screen_anchor(anchor)
        
        # Adjust based on which anchor
        if "right" in anchor.value:
            x -= self._width
        elif "center" in anchor.value and "left" not in anchor.value and "right" not in anchor.value:
            x -= self._width // 2
        
        if "bottom" in anchor.value:
            y -= self._height
        elif anchor == AnchorPoint.SCREEN_CENTER:
            y -= self._height // 2
        
        self._x = x
        self._y = y
        
        return self
    
    # Offset and padding
    
    def with_offset(self, x: int, y: int) -> 'RegionBuilder':
        """Add offset to current position.
        
        Args:
            x: X offset
            y: Y offset
            
        Returns:
            Self for chaining
        """
        self._offset_x = x
        self._offset_y = y
        self._x += x
        self._y += y
        return self
    
    def with_padding(self, padding: int) -> 'RegionBuilder':
        """Add padding (shrinks region inward).
        
        Args:
            padding: Padding in pixels
            
        Returns:
            Self for chaining
        """
        self._padding = padding
        self._x += padding
        self._y += padding
        self._width -= 2 * padding
        self._height -= 2 * padding
        return self
    
    def with_margin(self, margin: int) -> 'RegionBuilder':
        """Set margin for inside positioning.
        
        Args:
            margin: Margin in pixels
            
        Returns:
            Self for chaining
        """
        self._margin = margin
        return self
    
    # Size adjustments
    
    def grow(self, amount: int) -> 'RegionBuilder':
        """Grow region by amount in all directions.
        
        Args:
            amount: Pixels to grow
            
        Returns:
            Self for chaining
        """
        self._x -= amount
        self._y -= amount
        self._width += 2 * amount
        self._height += 2 * amount
        return self
    
    def shrink(self, amount: int) -> 'RegionBuilder':
        """Shrink region by amount in all directions.
        
        Args:
            amount: Pixels to shrink
            
        Returns:
            Self for chaining
        """
        self._x += amount
        self._y += amount
        self._width -= 2 * amount
        self._height -= 2 * amount
        return self
    
    def with_aspect_ratio(self, ratio: float) -> 'RegionBuilder':
        """Adjust size to maintain aspect ratio.
        
        Args:
            ratio: Width/height ratio
            
        Returns:
            Self for chaining
        """
        # Adjust width based on height
        self._width = int(self._height * ratio)
        return self
    
    # Building
    
    def build(self) -> Region:
        """Build the region with configured properties.
        
        Returns:
            Configured Region instance
        """
        # Ensure valid dimensions
        if self._width < 1:
            self._width = 1
        if self._height < 1:
            self._height = 1
        
        # Ensure non-negative position
        if self._x < 0:
            self._x = 0
        if self._y < 0:
            self._y = 0
        
        return Region(
            x=self._x,
            y=self._y,
            width=self._width,
            height=self._height,
            name=self._name
        )