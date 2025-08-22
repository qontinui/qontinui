"""Scroll options - ported from Qontinui framework.

Configuration for scroll actions.
"""

from enum import Enum, auto
from typing import Optional
from ...action_config import ActionConfig, ActionConfigBuilder


class ScrollDirection(Enum):
    """Direction of scroll action."""
    
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


class ScrollOptions(ActionConfig):
    """Configuration for scroll actions.
    
    Port of ScrollOptions from Qontinui framework.
    
    This class encapsulates all parameters needed to configure a scroll action,
    including direction, amount, and target location.
    """
    
    def __init__(self, builder: 'ScrollOptionsBuilder'):
        """Initialize ScrollOptions from builder.
        
        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self.direction = builder.direction
        self.clicks = builder.clicks
        self.smooth = builder.smooth
        self.delay_between_scrolls = builder.delay_between_scrolls
    
    def get_direction(self) -> ScrollDirection:
        """Get the scroll direction."""
        return self.direction
    
    def get_clicks(self) -> int:
        """Get the number of scroll clicks."""
        return self.clicks
    
    def is_smooth(self) -> bool:
        """Check if smooth scrolling is enabled."""
        return self.smooth
    
    def get_delay_between_scrolls(self) -> float:
        """Get the delay between scroll actions in seconds."""
        return self.delay_between_scrolls


class ScrollOptionsBuilder(ActionConfigBuilder['ScrollOptionsBuilder']):
    """Builder for constructing ScrollOptions with a fluent API.
    
    Port of ScrollOptions from Qontinui framework.Builder.
    """
    
    def __init__(self, original: Optional[ScrollOptions] = None):
        """Initialize builder.
        
        Args:
            original: Optional ScrollOptions instance to copy values from
        """
        super().__init__(original)
        
        if original:
            self.direction = original.direction
            self.clicks = original.clicks
            self.smooth = original.smooth
            self.delay_between_scrolls = original.delay_between_scrolls
        else:
            self.direction = ScrollDirection.DOWN
            self.clicks = 3
            self.smooth = False
            self.delay_between_scrolls = 0.1
    
    def set_direction(self, direction: ScrollDirection) -> 'ScrollOptionsBuilder':
        """Set the scroll direction.
        
        Args:
            direction: The direction to scroll
            
        Returns:
            This builder instance for chaining
        """
        self.direction = direction
        return self
    
    def set_clicks(self, clicks: int) -> 'ScrollOptionsBuilder':
        """Set the number of scroll clicks.
        
        Args:
            clicks: Number of scroll clicks to perform
            
        Returns:
            This builder instance for chaining
        """
        self.clicks = clicks
        return self
    
    def set_smooth(self, smooth: bool) -> 'ScrollOptionsBuilder':
        """Enable or disable smooth scrolling.
        
        Args:
            smooth: Whether to use smooth scrolling
            
        Returns:
            This builder instance for chaining
        """
        self.smooth = smooth
        return self
    
    def set_delay_between_scrolls(self, delay: float) -> 'ScrollOptionsBuilder':
        """Set the delay between scroll actions.
        
        Args:
            delay: Delay in seconds between scroll actions
            
        Returns:
            This builder instance for chaining
        """
        self.delay_between_scrolls = delay
        return self
    
    def build(self) -> ScrollOptions:
        """Build the immutable ScrollOptions object.
        
        Returns:
            A new instance of ScrollOptions
        """
        return ScrollOptions(self)
    
    def _self(self) -> 'ScrollOptionsBuilder':
        """Return self for fluent interface.
        
        Returns:
            This builder instance
        """
        return self