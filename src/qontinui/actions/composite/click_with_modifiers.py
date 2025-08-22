"""Convenience methods for click actions with keyboard modifiers.

These composite methods maintain Brobot's atomic action principle
by chaining KeyDown -> Click -> KeyUp actions.
"""

from typing import Any, Optional
from ..basic.click.click import Click, ClickOptions
from ..basic.type.key_down import KeyDown, KeyDownOptions
from ..basic.type.key_up import KeyUp, KeyUpOptions


class ClickWithModifiers:
    """Convenience methods for clicks with keyboard modifiers.
    
    These methods chain atomic actions to perform modifier+click
    combinations while maintaining Brobot's atomic action principle.
    """
    
    @staticmethod
    def shift_click(target: Any, options: Optional[ClickOptions] = None) -> bool:
        """Perform a click while holding the Shift key.
        
        Args:
            target: Target to click (Location, Region, StateObject, etc.)
            options: Optional click options
            
        Returns:
            True if successful
        """
        key_down = KeyDown(KeyDownOptions().add_key("SHIFT"))
        click = Click(options or ClickOptions())
        key_up = KeyUp(KeyUpOptions().add_key("SHIFT"))
        
        # Execute atomic actions in sequence
        if not key_down.execute():
            return False
        
        try:
            result = click.execute(target)
        finally:
            # Always release the key
            key_up.execute()
        
        return result
    
    @staticmethod
    def ctrl_click(target: Any, options: Optional[ClickOptions] = None) -> bool:
        """Perform a click while holding the Ctrl key.
        
        Args:
            target: Target to click (Location, Region, StateObject, etc.)
            options: Optional click options
            
        Returns:
            True if successful
        """
        key_down = KeyDown(KeyDownOptions().add_key("CTRL"))
        click = Click(options or ClickOptions())
        key_up = KeyUp(KeyUpOptions().add_key("CTRL"))
        
        # Execute atomic actions in sequence
        if not key_down.execute():
            return False
        
        try:
            result = click.execute(target)
        finally:
            # Always release the key
            key_up.execute()
        
        return result
    
    @staticmethod
    def alt_click(target: Any, options: Optional[ClickOptions] = None) -> bool:
        """Perform a click while holding the Alt key.
        
        Args:
            target: Target to click (Location, Region, StateObject, etc.)
            options: Optional click options
            
        Returns:
            True if successful
        """
        key_down = KeyDown(KeyDownOptions().add_key("ALT"))
        click = Click(options or ClickOptions())
        key_up = KeyUp(KeyUpOptions().add_key("ALT"))
        
        # Execute atomic actions in sequence
        if not key_down.execute():
            return False
        
        try:
            result = click.execute(target)
        finally:
            # Always release the key
            key_up.execute()
        
        return result
    
    @staticmethod
    def ctrl_shift_click(target: Any, options: Optional[ClickOptions] = None) -> bool:
        """Perform a click while holding Ctrl and Shift keys.
        
        Args:
            target: Target to click (Location, Region, StateObject, etc.)
            options: Optional click options
            
        Returns:
            True if successful
        """
        key_down = KeyDown(KeyDownOptions().add_key("CTRL").add_key("SHIFT"))
        click = Click(options or ClickOptions())
        key_up = KeyUp(KeyUpOptions().add_key("SHIFT").add_key("CTRL"))
        
        # Execute atomic actions in sequence
        if not key_down.execute():
            return False
        
        try:
            result = click.execute(target)
        finally:
            # Always release keys in reverse order
            key_up.execute()
        
        return result
    
    @staticmethod
    def meta_click(target: Any, options: Optional[ClickOptions] = None) -> bool:
        """Perform a click while holding the Meta/Windows/Command key.
        
        Args:
            target: Target to click (Location, Region, StateObject, etc.)
            options: Optional click options
            
        Returns:
            True if successful
        """
        key_down = KeyDown(KeyDownOptions().add_key("META"))
        click = Click(options or ClickOptions())
        key_up = KeyUp(KeyUpOptions().add_key("META"))
        
        # Execute atomic actions in sequence
        if not key_down.execute():
            return False
        
        try:
            result = click.execute(target)
        finally:
            # Always release the key
            key_up.execute()
        
        return result


class FluentClickWithModifiers:
    """Fluent interface for building modifier+click combinations.
    
    Allows chaining modifiers before executing the click action.
    """
    
    def __init__(self, target: Any, options: Optional[ClickOptions] = None):
        """Initialize fluent click builder.
        
        Args:
            target: Target to click
            options: Optional click options
        """
        self.target = target
        self.options = options or ClickOptions()
        self.modifiers = []
    
    def with_shift(self) -> 'FluentClickWithModifiers':
        """Add Shift modifier.
        
        Returns:
            Self for fluent interface
        """
        if "SHIFT" not in self.modifiers:
            self.modifiers.append("SHIFT")
        return self
    
    def with_ctrl(self) -> 'FluentClickWithModifiers':
        """Add Ctrl modifier.
        
        Returns:
            Self for fluent interface
        """
        if "CTRL" not in self.modifiers:
            self.modifiers.append("CTRL")
        return self
    
    def with_alt(self) -> 'FluentClickWithModifiers':
        """Add Alt modifier.
        
        Returns:
            Self for fluent interface
        """
        if "ALT" not in self.modifiers:
            self.modifiers.append("ALT")
        return self
    
    def with_meta(self) -> 'FluentClickWithModifiers':
        """Add Meta/Windows/Command modifier.
        
        Returns:
            Self for fluent interface
        """
        if "META" not in self.modifiers:
            self.modifiers.append("META")
        return self
    
    def execute(self) -> bool:
        """Execute the click with all specified modifiers.
        
        Returns:
            True if successful
        """
        if not self.modifiers:
            # No modifiers, just do a regular click
            return Click(self.options).execute(self.target)
        
        # Create key down action with all modifiers
        key_down_options = KeyDownOptions()
        for modifier in self.modifiers:
            key_down_options.add_key(modifier)
        key_down = KeyDown(key_down_options)
        
        # Create key up action with modifiers in reverse order
        key_up_options = KeyUpOptions()
        for modifier in reversed(self.modifiers):
            key_up_options.add_key(modifier)
        key_up = KeyUp(key_up_options)
        
        # Execute atomic actions in sequence
        if not key_down.execute():
            return False
        
        try:
            result = Click(self.options).execute(self.target)
        finally:
            # Always release the keys
            key_up.execute()
        
        return result


def modified_click(target: Any, options: Optional[ClickOptions] = None) -> FluentClickWithModifiers:
    """Create a fluent builder for modifier+click combinations.
    
    Example usage:
        # Ctrl+Shift+Click
        modified_click(location).with_ctrl().with_shift().execute()
        
        # Alt+Click with double click
        modified_click(region, ClickOptions().double()).with_alt().execute()
    
    Args:
        target: Target to click
        options: Optional click options
        
    Returns:
        Fluent builder for chaining modifiers
    """
    return FluentClickWithModifiers(target, options)