"""Key up options - ported from Qontinui framework.

Configuration for keyboard key release operations.
"""

from dataclasses import dataclass, field
from typing import List
from ...action_config import ActionConfig


@dataclass
class KeyUpOptions(ActionConfig):
    """Configuration for KeyUp actions.
    
    Port of KeyUpOptions from Qontinui framework class.
    
    Configures keyboard key release operations to complete
    key press sequences started with KeyDown.
    """
    
    # Key release configuration
    keys: List[str] = field(default_factory=list)  # Keys to release
    modifiers: List[str] = field(default_factory=list)  # Modifier keys to release
    pause_between_keys: float = 0.1  # Delay between releasing multiple keys
    release_modifiers_first: bool = False  # Whether to release modifiers before other keys
    
    def add_key(self, key: str) -> 'KeyUpOptions':
        """Add a key to release.
        
        Args:
            key: Key to add
            
        Returns:
            Self for fluent interface
        """
        self.keys.append(key)
        return self
    
    def add_keys(self, *keys: str) -> 'KeyUpOptions':
        """Add multiple keys to release.
        
        Args:
            *keys: Keys to add
            
        Returns:
            Self for fluent interface
        """
        self.keys.extend(keys)
        return self
    
    def with_modifiers(self, *modifiers: str) -> 'KeyUpOptions':
        """Set modifier keys to release.
        
        Args:
            *modifiers: Modifier keys (ctrl, shift, alt, cmd)
            
        Returns:
            Self for fluent interface
        """
        self.modifiers = list(modifiers)
        return self
    
    def with_pause(self, seconds: float) -> 'KeyUpOptions':
        """Set pause between key releases.
        
        Args:
            seconds: Pause duration
            
        Returns:
            Self for fluent interface
        """
        self.pause_between_keys = seconds
        return self
    
    def release_modifiers_before_keys(self, value: bool = True) -> 'KeyUpOptions':
        """Set whether to release modifiers first.
        
        Args:
            value: True to release modifiers first
            
        Returns:
            Self for fluent interface
        """
        self.release_modifiers_first = value
        return self