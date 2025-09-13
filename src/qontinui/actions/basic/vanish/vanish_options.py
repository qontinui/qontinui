"""Vanish options - ported from Qontinui framework.

Configuration for vanish actions.
"""

from typing import Optional
from ...action_config import ActionConfig, ActionConfigBuilder


class VanishOptions(ActionConfig):
    """Configuration for vanish actions.
    
    Port of VanishOptions from Qontinui framework.
    
    The Vanish action waits for visual elements to disappear from the screen.
    This is useful for waiting for loading screens, popups, or other temporary
    UI elements to go away before proceeding with automation.
    """
    
    def __init__(self, builder: 'VanishOptionsBuilder'):
        """Initialize VanishOptions from builder.
        
        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self.max_wait_time = builder.max_wait_time
        self.poll_interval = builder.poll_interval
    
    def get_max_wait_time(self) -> float:
        """Get the maximum time to wait for element to vanish."""
        return self.max_wait_time
    
    def get_poll_interval(self) -> float:
        """Get the interval between vanish checks."""
        return self.poll_interval


class VanishOptionsBuilder(ActionConfigBuilder):
    """Builder for constructing VanishOptions with a fluent API.
    
    Port of VanishOptions from Qontinui framework.Builder.
    """
    
    def __init__(self, original: Optional[VanishOptions] = None):
        """Initialize builder.
        
        Args:
            original: Optional VanishOptions instance to copy values from
        """
        super().__init__(original)
        
        if original:
            self.max_wait_time = original.max_wait_time
            self.poll_interval = original.poll_interval
        else:
            self.max_wait_time = 10.0  # Default 10 seconds max wait
            self.poll_interval = 0.5    # Check every 0.5 seconds
    
    def set_max_wait_time(self, seconds: float) -> 'VanishOptionsBuilder':
        """Set the maximum time to wait for vanish.
        
        Args:
            seconds: Maximum wait time in seconds
            
        Returns:
            This builder instance for chaining
        """
        self.max_wait_time = seconds
        return self
    
    def set_poll_interval(self, seconds: float) -> 'VanishOptionsBuilder':
        """Set the polling interval.
        
        Args:
            seconds: Time between vanish checks in seconds
            
        Returns:
            This builder instance for chaining
        """
        self.poll_interval = seconds
        return self
    
    def build(self) -> VanishOptions:
        """Build the immutable VanishOptions object.
        
        Returns:
            A new instance of VanishOptions
        """
        return VanishOptions(self)
    
    def _self(self) -> 'VanishOptionsBuilder':
        """Return self for fluent interface.
        
        Returns:
            This builder instance
        """
        return self