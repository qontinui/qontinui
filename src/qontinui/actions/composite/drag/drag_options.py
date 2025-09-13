"""Drag options - ported from Qontinui framework.

Configuration for drag actions.
"""

from typing import Optional
from ...action_config import ActionConfig, ActionConfigBuilder
from ...basic.mouse.mouse_press_options import MousePressOptions, MouseButton


class DragOptions(ActionConfig):
    """Configuration for drag actions.
    
    Port of DragOptions from Qontinui framework.
    
    DragOptions configures a drag operation which is implemented as a chain of 6 actions:
    Find source → Find target → MouseMove to source → MouseDown → MouseMove to target → MouseUp
    """
    
    def __init__(self, builder: 'DragOptionsBuilder'):
        """Initialize DragOptions from builder.
        
        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self.mouse_press_options = builder.mouse_press_options
        self.delay_between_mouse_down_and_move = builder.delay_between_mouse_down_and_move
        self.delay_after_drag = builder.delay_after_drag
    
    def get_mouse_press_options(self) -> MousePressOptions:
        """Get mouse press options."""
        return self.mouse_press_options
    
    def get_delay_between_mouse_down_and_move(self) -> float:
        """Get delay between mouse down and move."""
        return self.delay_between_mouse_down_and_move
    
    def get_delay_after_drag(self) -> float:
        """Get delay after drag."""
        return self.delay_after_drag


class DragOptionsBuilder(ActionConfigBuilder):
    """Builder for constructing DragOptions with a fluent API.
    
    Port of DragOptions from Qontinui framework.Builder.
    """
    
    def __init__(self, original: Optional[DragOptions] = None):
        """Initialize builder.
        
        Args:
            original: Optional DragOptions instance to copy values from
        """
        super().__init__(original)
        
        if original:
            self.mouse_press_options = original.mouse_press_options
            self.delay_between_mouse_down_and_move = original.delay_between_mouse_down_and_move
            self.delay_after_drag = original.delay_after_drag
        else:
            self.mouse_press_options = (MousePressOptions.builder()
                                       .set_button(MouseButton.LEFT)
                                       .build())
            self.delay_between_mouse_down_and_move = 0.5
            self.delay_after_drag = 0.5
    
    def set_mouse_press_options(self, mouse_press_options: MousePressOptions) -> 'DragOptionsBuilder':
        """Set the mouse press options for the drag operation.
        
        Args:
            mouse_press_options: Configuration for mouse button and timing
            
        Returns:
            This builder instance for chaining
        """
        self.mouse_press_options = mouse_press_options
        return self
    
    def set_delay_between_mouse_down_and_move(self, seconds: float) -> 'DragOptionsBuilder':
        """Set the delay between mouse down and the drag movement.
        
        Args:
            seconds: Delay in seconds
            
        Returns:
            This builder instance for chaining
        """
        self.delay_between_mouse_down_and_move = seconds
        return self
    
    def set_delay_after_drag(self, seconds: float) -> 'DragOptionsBuilder':
        """Set the delay after completing the drag operation.
        
        Args:
            seconds: Delay in seconds
            
        Returns:
            This builder instance for chaining
        """
        self.delay_after_drag = seconds
        return self
    
    def build(self) -> DragOptions:
        """Build the immutable DragOptions object.
        
        Returns:
            A new instance of DragOptions
        """
        return DragOptions(self)
    
    def _self(self) -> 'DragOptionsBuilder':
        """Return self for fluent interface.
        
        Returns:
            This builder instance
        """
        return self