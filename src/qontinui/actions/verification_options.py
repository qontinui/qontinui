"""Verification options - ported from Qontinui framework.

Options for verifying action results.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List


class Event(Enum):
    """Verification event types."""
    
    TEXT_APPEARS = auto()
    """Verify that text appears"""
    
    TEXT_DISAPPEARS = auto()
    """Verify that text disappears"""
    
    IMAGE_APPEARS = auto()
    """Verify that image appears"""
    
    IMAGE_DISAPPEARS = auto()
    """Verify that image disappears"""
    
    STATE_CHANGE = auto()
    """Verify that state changes"""
    
    NONE = auto()
    """No verification"""


@dataclass
class VerificationOptions:
    """Options for verifying action results.
    
    Port of VerificationOptions from Qontinui framework.
    
    This class defines conditions that determine when an action
    should stop repeating or be considered successful.
    """
    
    event: Event = Event.NONE
    """The verification event type"""
    
    text: str = ""
    """Text to verify for text events"""
    
    images: List[str] = None
    """Images to verify for image events"""
    
    timeout: float = 5.0
    """Maximum time to wait for verification"""
    
    @classmethod
    def builder(cls) -> 'VerificationOptionsBuilder':
        """Create a builder for VerificationOptions.
        
        Returns:
            A new builder instance
        """
        return VerificationOptionsBuilder()
    
    def to_builder(self) -> 'VerificationOptionsBuilder':
        """Convert this instance to a builder for modification.
        
        Returns:
            A builder pre-populated with this instance's values
        """
        builder = VerificationOptionsBuilder()
        builder.event = self.event
        builder.text = self.text
        builder.images = self.images.copy() if self.images else []
        builder.timeout = self.timeout
        return builder


class VerificationOptionsBuilder:
    """Builder for VerificationOptions."""
    
    def __init__(self):
        self.event = Event.NONE
        self.text = ""
        self.images = []
        self.timeout = 5.0
    
    def set_event(self, event: Event) -> 'VerificationOptionsBuilder':
        """Set the verification event type."""
        self.event = event
        return self
    
    def set_text(self, text: str) -> 'VerificationOptionsBuilder':
        """Set text to verify."""
        self.text = text
        return self
    
    def set_images(self, images: List[str]) -> 'VerificationOptionsBuilder':
        """Set images to verify."""
        self.images = images
        return self
    
    def set_timeout(self, timeout: float) -> 'VerificationOptionsBuilder':
        """Set verification timeout."""
        self.timeout = timeout
        return self
    
    def build(self) -> VerificationOptions:
        """Build the VerificationOptions instance.
        
        Returns:
            A new VerificationOptions with the configured values
        """
        return VerificationOptions(
            event=self.event,
            text=self.text,
            images=self.images.copy() if self.images else None,
            timeout=self.timeout
        )