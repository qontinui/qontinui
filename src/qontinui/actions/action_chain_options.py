"""Action chain options - ported from Qontinui framework.

Configuration for executing a chain of actions with specific chaining behavior.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional
from .action_config import ActionConfig, ActionConfigBuilder


class ChainingStrategy(Enum):
    """Defines how results from one action in the chain relate to the next."""
    
    NESTED = auto()
    """Each action searches within the results of the previous action.
    Best for hierarchical searches like finding a button within a dialog."""
    
    CONFIRM = auto()
    """Each action validates/confirms the results of the previous action.
    Best for eliminating false positives by requiring multiple confirmations."""


class ActionChainOptions(ActionConfig):
    """Configuration for executing a chain of actions with specific chaining behavior.
    
    Port of ActionChainOptions from Qontinui framework.
    
    ActionChainOptions wraps an initial ActionConfig and adds parameters that control
    how an entire sequence of actions behaves. This includes the chaining strategy
    (nested vs. confirmed) and the list of subsequent actions to execute.
    
    This design separates the configuration of individual actions from the configuration
    of how those actions work together, following the Single Responsibility Principle.
    """
    
    def __init__(self, builder: 'ActionChainOptionsBuilder'):
        """Initialize ActionChainOptions from builder.
        
        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self.initial_action = builder.initial_action
        self.strategy = builder.strategy
        self.chained_actions = list(builder.chained_actions)
    
    def get_initial_action(self) -> ActionConfig:
        """Get the initial action in the chain."""
        return self.initial_action
    
    def get_strategy(self) -> ChainingStrategy:
        """Get the chaining strategy."""
        return self.strategy
    
    def get_chained_actions(self) -> List[ActionConfig]:
        """Return an unmodifiable view of the chained actions list.
        
        Returns:
            List of chained actions
        """
        return list(self.chained_actions)  # Return a copy to prevent modification


class ActionChainOptionsBuilder(ActionConfigBuilder['ActionChainOptionsBuilder']):
    """Builder for constructing ActionChainOptions with a fluent API.
    
    Port of ActionChainOptions from Qontinui framework.Builder.
    """
    
    def __init__(self, initial_action: ActionConfig):
        """Create a new Builder with the initial action.
        
        Args:
            initial_action: The first action in the chain
        """
        super().__init__()
        self.initial_action = initial_action
        self.strategy = ChainingStrategy.NESTED
        self.chained_actions: List[ActionConfig] = []
    
    def set_strategy(self, strategy: ChainingStrategy) -> 'ActionChainOptionsBuilder':
        """Set the chaining strategy.
        
        Args:
            strategy: How actions in the chain relate to each other
            
        Returns:
            This builder instance for chaining
        """
        self.strategy = strategy
        return self
    
    def then(self, action: ActionConfig) -> 'ActionChainOptionsBuilder':
        """Add an action to the chain.
        
        Args:
            action: The action to add to the chain
            
        Returns:
            This builder instance for chaining
        """
        self.chained_actions.append(action)
        return self
    
    def build(self) -> ActionChainOptions:
        """Build the immutable ActionChainOptions object.
        
        Returns:
            A new instance of ActionChainOptions
        """
        return ActionChainOptions(self)
    
    def _self(self) -> 'ActionChainOptionsBuilder':
        """Return self for fluent interface.
        
        Returns:
            This builder instance
        """
        return self