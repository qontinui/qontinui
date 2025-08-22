"""StateText class - ported from Qontinui framework.

Text clues for faster state searching.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class StateText:
    """Text clue for faster state searching.
    
    Port of StateText from Qontinui framework class.
    
    StateText represents text that should be present when a state is active.
    This is different from StateString (which is for text input fields).
    StateText is used for quickly identifying states by looking for specific
    text on screen, which is often faster than image matching.
    """
    
    text: str = ""  # The text to search for
    owner_state_id: Optional[int] = None  # ID of state that owns this text
    
    # Search configuration
    search_regions: List['Region'] = field(default_factory=list)  # Where to look for text
    case_sensitive: bool = False  # Whether search is case-sensitive
    exact_match: bool = False  # Require exact match vs contains
    
    # OCR configuration
    language: str = "eng"  # OCR language
    min_confidence: float = 0.7  # Minimum OCR confidence
    
    # Metadata
    description: str = ""  # Human-readable description
    
    def __post_init__(self):
        """Validate text configuration."""
        if not self.text:
            raise ValueError("StateText must have non-empty text")
    
    def matches(self, found_text: str) -> bool:
        """Check if found text matches this StateText.
        
        Args:
            found_text: Text found on screen
            
        Returns:
            True if text matches based on configuration
        """
        if not found_text:
            return False
            
        # Prepare texts for comparison
        search_text = self.text
        compare_text = found_text
        
        if not self.case_sensitive:
            search_text = search_text.lower()
            compare_text = compare_text.lower()
        
        # Check match type
        if self.exact_match:
            return search_text == compare_text
        else:
            return search_text in compare_text
    
    def with_region(self, region: 'Region') -> 'StateText':
        """Add a search region.
        
        Args:
            region: Region to search in
            
        Returns:
            Self for fluent interface
        """
        self.search_regions.append(region)
        return self
    
    def case_sensitive(self, sensitive: bool = True) -> 'StateText':
        """Set case sensitivity.
        
        Args:
            sensitive: Whether to be case-sensitive
            
        Returns:
            Self for fluent interface
        """
        self.case_sensitive = sensitive
        return self
    
    def exact(self, exact: bool = True) -> 'StateText':
        """Require exact match.
        
        Args:
            exact: Whether to require exact match
            
        Returns:
            Self for fluent interface
        """
        self.exact_match = exact
        return self
    
    def with_confidence(self, confidence: float) -> 'StateText':
        """Set minimum OCR confidence.
        
        Args:
            confidence: Minimum confidence (0.0-1.0)
            
        Returns:
            Self for fluent interface
        """
        self.min_confidence = confidence
        return self
    
    def __hash__(self) -> int:
        """Hash based on text and owner.
        
        Returns:
            Hash value
        """
        return hash((self.text, self.owner_state_id))
    
    def __eq__(self, other) -> bool:
        """Equality based on text and configuration.
        
        Args:
            other: Other StateText to compare
            
        Returns:
            True if equal
        """
        if not isinstance(other, StateText):
            return False
        return (self.text == other.text and 
                self.owner_state_id == other.owner_state_id and
                self.case_sensitive == other.case_sensitive and
                self.exact_match == other.exact_match)
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            Description of StateText
        """
        parts = [f"'{self.text}'"]
        if self.owner_state_id:
            parts.append(f"state={self.owner_state_id}")
        if self.case_sensitive:
            parts.append("case-sensitive")
        if self.exact_match:
            parts.append("exact")
        return f"StateText({', '.join(parts)})"