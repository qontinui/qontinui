"""Key down options - ported from Qontinui framework.

Configuration for keyboard key press-and-hold operations.
"""

from dataclasses import dataclass, field
from typing import List
from ..action_config import ActionConfig


@dataclass
class KeyDownOptions(ActionConfig):
    """Configuration for KeyDown actions.
    
    Port of KeyDownOptions from Qontinui framework class.
    
    Configures keyboard key press operations where keys are
    held down without releasing.
    """
    
    # Key press configuration
    keys: List[str] = field(default_factory=list)  # Keys to press
    modifiers: List[str] = field(default_factory=list)  # Modifier keys (Ctrl, Shift, Alt)
    pause_between_keys: float = 0.1  # Delay between pressing multiple keys
    
    def add_key(self, key: str) -> 'KeyDownOptions':
        """Add a key to press.
        
        Args:
            key: Key to add
            
        Returns:
            Self for fluent interface
        """
        self.keys.append(key)
        return self
    
    def add_keys(self, *keys: str) -> 'KeyDownOptions':
        """Add multiple keys to press.
        
        Args:
            *keys: Keys to add
            
        Returns:
            Self for fluent interface
        """
        self.keys.extend(keys)
        return self
    
    def with_modifiers(self, *modifiers: str) -> 'KeyDownOptions':
        """Set modifier keys.
        
        Args:
            *modifiers: Modifier keys (ctrl, shift, alt, cmd)
            
        Returns:
            Self for fluent interface
        """
        self.modifiers = list(modifiers)
        return self
    
    def with_pause(self, seconds: float) -> 'KeyDownOptions':
        """Set pause between keys.
        
        Args:
            seconds: Pause duration
            
        Returns:
            Self for fluent interface
        """
        self.pause_between_keys = seconds
        return self