"""Repetition options - ported from Qontinui framework.

Options for repeating actions.
"""

from dataclasses import dataclass


@dataclass
class RepetitionOptions:
    """Options for controlling how actions are repeated.
    
    Port of RepetitionOptions from Qontinui framework.
    
    This class encapsulates parameters for controlling the repetition
    of actions on individual targets and the timing between repetitions.
    """
    
    times_to_repeat_individual_action: int = 1
    """Number of times to repeat an action on an individual target"""
    
    pause_between_individual_actions: float = 0.0
    """Pause duration between individual actions in seconds"""
    
    max_repetitions: int = 10
    """Maximum number of repetitions before giving up"""
    
    @classmethod
    def builder(cls) -> 'RepetitionOptionsBuilder':
        """Create a builder for RepetitionOptions.
        
        Returns:
            A new builder instance
        """
        return RepetitionOptionsBuilder()
    
    def to_builder(self) -> 'RepetitionOptionsBuilder':
        """Convert this instance to a builder for modification.
        
        Returns:
            A builder pre-populated with this instance's values
        """
        builder = RepetitionOptionsBuilder()
        builder.times_to_repeat_individual_action = self.times_to_repeat_individual_action
        builder.pause_between_individual_actions = self.pause_between_individual_actions
        builder.max_repetitions = self.max_repetitions
        return builder
    
    def get_times_to_repeat_individual_action(self) -> int:
        """Get the number of times to repeat an action on an individual target."""
        return self.times_to_repeat_individual_action
    
    def get_pause_between_individual_actions(self) -> float:
        """Get the pause duration between individual actions in seconds."""
        return self.pause_between_individual_actions


class RepetitionOptionsBuilder:
    """Builder for RepetitionOptions."""
    
    def __init__(self):
        self.times_to_repeat_individual_action = 1
        self.pause_between_individual_actions = 0.0
        self.max_repetitions = 10
    
    def set_times_to_repeat_individual_action(self, times: int) -> 'RepetitionOptionsBuilder':
        """Set number of times to repeat action on individual target."""
        self.times_to_repeat_individual_action = max(1, times)
        return self
    
    def set_pause_between_individual_actions(self, pause: float) -> 'RepetitionOptionsBuilder':
        """Set pause duration between individual actions."""
        self.pause_between_individual_actions = max(0.0, pause)
        return self
    
    def set_max_repetitions(self, max_reps: int) -> 'RepetitionOptionsBuilder':
        """Set maximum number of repetitions."""
        self.max_repetitions = max(1, max_reps)
        return self
    
    def build(self) -> RepetitionOptions:
        """Build the RepetitionOptions instance.
        
        Returns:
            A new RepetitionOptions with the configured values
        """
        return RepetitionOptions(
            times_to_repeat_individual_action=self.times_to_repeat_individual_action,
            pause_between_individual_actions=self.pause_between_individual_actions,
            max_repetitions=self.max_repetitions
        )