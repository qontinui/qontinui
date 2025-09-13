"""Click options - ported from Qontinui framework.

Configuration for all Click actions.
"""

from typing import Optional
from ...action_config import ActionConfig, ActionConfigBuilder
from ...verification_options import VerificationOptions, VerificationOptionsBuilder
from ...repetition_options import RepetitionOptions, RepetitionOptionsBuilder
from ..mouse.mouse_press_options import MousePressOptions, MousePressOptionsBuilder


class ClickOptions(ActionConfig):
    """Configuration for all Click actions.
    
    Port of ClickOptions from Qontinui framework.
    
    This class encapsulates all parameters for performing mouse clicks, including the
    click type and any verification conditions that should terminate a repeating click.
    It is an immutable object and must be constructed using its inner Builder.
    
    This specialized configuration enhances API clarity by only exposing options
    relevant to click operations.
    
    Example usage:
        click_until_text_appears = ClickOptionsBuilder()
            .set_number_of_clicks(2)
            .set_press_options(MousePressOptions.builder()
                .set_button(MouseButton.LEFT)
                .build())
            .set_verification(VerificationOptions.builder()
                .set_event(Event.TEXT_APPEARS)
                .set_text("Success"))
            .build()
    """
    
    def __init__(self, builder: 'ClickOptionsBuilder'):
        """Initialize ClickOptions from builder.
        
        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self.number_of_clicks = builder.number_of_clicks
        self.mouse_press_options = builder.mouse_press_options
        self.verification_options = builder.verification_options.build()
        self.repetition_options = builder.repetition_options.build()
    
    def get_number_of_clicks(self) -> int:
        """Get the number of clicks to perform."""
        return self.number_of_clicks
    
    def get_mouse_press_options(self) -> MousePressOptions:
        """Get mouse press options."""
        return self.mouse_press_options
    
    def get_verification_options(self) -> VerificationOptions:
        """Get verification options."""
        return self.verification_options
    
    def get_repetition_options(self) -> RepetitionOptions:
        """Get repetition options."""
        return self.repetition_options
    
    def get_times_to_repeat_individual_action(self) -> int:
        """Convenience getter for the number of times to repeat an action on an individual target.
        
        Returns:
            The number of repetitions for an individual action
        """
        return self.repetition_options.get_times_to_repeat_individual_action()
    
    def get_pause_between_individual_actions(self) -> float:
        """Convenience getter for the pause between individual actions.
        
        Returns:
            The pause duration between individual actions in seconds
        """
        return self.repetition_options.get_pause_between_individual_actions()


class ClickOptionsBuilder(ActionConfigBuilder):
    """Builder for constructing ClickOptions with a fluent API.
    
    Port of ClickOptions from Qontinui framework.Builder.
    """
    
    def __init__(self, original: Optional[ClickOptions] = None):
        """Initialize builder.
        
        Args:
            original: Optional ClickOptions instance to copy values from
        """
        super().__init__(original)
        
        if original:
            self.number_of_clicks = original.number_of_clicks
            self.mouse_press_options = original.mouse_press_options.to_builder().build()
            self.verification_options = original.verification_options.to_builder()
            self.repetition_options = original.repetition_options.to_builder()
        else:
            self.number_of_clicks = 1
            self.mouse_press_options = MousePressOptions.builder().build()
            self.verification_options = VerificationOptions.builder()
            self.repetition_options = RepetitionOptions.builder()
    
    def set_number_of_clicks(self, number_of_clicks: int) -> 'ClickOptionsBuilder':
        """Set the number of times to click. For example, 2 for a double-click.
        
        Args:
            number_of_clicks: The number of clicks to perform
            
        Returns:
            This builder instance for chaining
        """
        self.number_of_clicks = max(1, number_of_clicks)  # Ensure at least 1 click
        return self
    
    def set_press_options(self, press_options: MousePressOptions) -> 'ClickOptionsBuilder':
        """Configure the pause behaviors for the press-and-release part of the click.
        
        Args:
            press_options: The mouse press options
            
        Returns:
            This builder instance for chaining
        """
        self.mouse_press_options = press_options
        return self
    
    def set_verification(self, verification_options_builder: VerificationOptionsBuilder) -> 'ClickOptionsBuilder':
        """Set the verification conditions that determine when this click action should stop repeating.
        
        Args:
            verification_options_builder: The builder for the verification options
            
        Returns:
            This builder instance for chaining
        """
        self.verification_options = verification_options_builder
        return self
    
    def set_repetition(self, repetition_options_builder: RepetitionOptionsBuilder) -> 'ClickOptionsBuilder':
        """Set the repetition options for controlling how clicks are repeated.
        
        Args:
            repetition_options_builder: The builder for the repetition options
            
        Returns:
            This builder instance for chaining
        """
        self.repetition_options = repetition_options_builder
        return self
    
    def build(self) -> ClickOptions:
        """Build the immutable ClickOptions object.
        
        Returns:
            A new instance of ClickOptions
        """
        return ClickOptions(self)
    
    def _self(self) -> 'ClickOptionsBuilder':
        """Return self for fluent interface.
        
        Returns:
            This builder instance
        """
        return self