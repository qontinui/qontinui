"""Define region options - ported from Qontinui framework.

Configuration for region definition actions.
"""

from dataclasses import dataclass
from enum import Enum, auto
from ...action_config import ActionConfig


class DefineAs(Enum):
    """How to define a region.
    
    Port of DefineAs from Qontinui framework enum.
    """
    FOCUSED_WINDOW = auto()  # Define as active window bounds
    MATCH = auto()  # Define as match bounds
    BELOW_MATCH = auto()  # Define below a match
    ABOVE_MATCH = auto()  # Define above a match
    LEFT_OF_MATCH = auto()  # Define to left of match
    RIGHT_OF_MATCH = auto()  # Define to right of match
    INSIDE_ANCHORS = auto()  # Smallest region containing all anchors
    OUTSIDE_ANCHORS = auto()  # Largest region containing all anchors
    INCLUDING_MATCHES = auto()  # Region including all matches


@dataclass
class DefineRegionOptions(ActionConfig):
    """Configuration for region definition actions.
    
    Port of DefineRegionOptions from Qontinui framework class.
    
    Configures how to define regions based on various strategies.
    """
    
    # Definition configuration
    define_as: DefineAs = DefineAs.MATCH
    offset_x: int = 0  # Horizontal offset from base position
    offset_y: int = 0  # Vertical offset from base position
    expand_width: int = 0  # Expand region width
    expand_height: int = 0  # Expand region height
    
    def as_window(self) -> 'DefineRegionOptions':
        """Define as focused window.
        
        Returns:
            Self for fluent interface
        """
        self.define_as = DefineAs.FOCUSED_WINDOW
        return self
    
    def as_match(self) -> 'DefineRegionOptions':
        """Define as match bounds.
        
        Returns:
            Self for fluent interface
        """
        self.define_as = DefineAs.MATCH
        return self
    
    def below_match(self) -> 'DefineRegionOptions':
        """Define below match.
        
        Returns:
            Self for fluent interface
        """
        self.define_as = DefineAs.BELOW_MATCH
        return self
    
    def above_match(self) -> 'DefineRegionOptions':
        """Define above match.
        
        Returns:
            Self for fluent interface
        """
        self.define_as = DefineAs.ABOVE_MATCH
        return self
    
    def left_of_match(self) -> 'DefineRegionOptions':
        """Define to left of match.
        
        Returns:
            Self for fluent interface
        """
        self.define_as = DefineAs.LEFT_OF_MATCH
        return self
    
    def right_of_match(self) -> 'DefineRegionOptions':
        """Define to right of match.
        
        Returns:
            Self for fluent interface
        """
        self.define_as = DefineAs.RIGHT_OF_MATCH
        return self
    
    def inside_anchors(self) -> 'DefineRegionOptions':
        """Define as smallest region containing anchors.
        
        Returns:
            Self for fluent interface
        """
        self.define_as = DefineAs.INSIDE_ANCHORS
        return self
    
    def outside_anchors(self) -> 'DefineRegionOptions':
        """Define as largest region containing anchors.
        
        Returns:
            Self for fluent interface
        """
        self.define_as = DefineAs.OUTSIDE_ANCHORS
        return self
    
    def including_matches(self) -> 'DefineRegionOptions':
        """Define as region including all matches.
        
        Returns:
            Self for fluent interface
        """
        self.define_as = DefineAs.INCLUDING_MATCHES
        return self
    
    def with_offset(self, x: int, y: int) -> 'DefineRegionOptions':
        """Set region offset.
        
        Args:
            x: Horizontal offset
            y: Vertical offset
            
        Returns:
            Self for fluent interface
        """
        self.offset_x = x
        self.offset_y = y
        return self
    
    def with_expansion(self, width: int, height: int) -> 'DefineRegionOptions':
        """Set region expansion.
        
        Args:
            width: Width expansion
            height: Height expansion
            
        Returns:
            Self for fluent interface
        """
        self.expand_width = width
        self.expand_height = height
        return self