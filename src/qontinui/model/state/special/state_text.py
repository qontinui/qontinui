"""StateText special state - ported from Qontinui framework.

Special state variant focused on text-based state identification.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from enum import Enum, auto
import re
from ..state import State
from ..state_enum import StateEnum
from ..state_string import StateString
from .special_state_type import SpecialStateType


class TextMatchType(Enum):
    """How to match text in states."""
    EXACT = auto()  # Exact match
    CONTAINS = auto()  # Contains substring
    REGEX = auto()  # Regular expression match
    FUZZY = auto()  # Fuzzy/approximate match
    STARTS_WITH = auto()  # Starts with string
    ENDS_WITH = auto()  # Ends with string


@dataclass
class TextPattern:
    """Pattern for matching text in states."""
    pattern: str
    match_type: TextMatchType = TextMatchType.CONTAINS
    case_sensitive: bool = False
    confidence: float = 0.8  # For fuzzy matching
    regex_flags: int = 0  # For regex matching
    
    def matches(self, text: str) -> bool:
        """Check if text matches this pattern.
        
        Args:
            text: Text to check
            
        Returns:
            True if matches
        """
        if not self.case_sensitive:
            text = text.lower()
            pattern = self.pattern.lower()
        else:
            pattern = self.pattern
        
        if self.match_type == TextMatchType.EXACT:
            return text == pattern
        elif self.match_type == TextMatchType.CONTAINS:
            return pattern in text
        elif self.match_type == TextMatchType.STARTS_WITH:
            return text.startswith(pattern)
        elif self.match_type == TextMatchType.ENDS_WITH:
            return text.endswith(pattern)
        elif self.match_type == TextMatchType.REGEX:
            try:
                return re.search(self.pattern, text, self.regex_flags) is not None
            except re.error:
                return False
        elif self.match_type == TextMatchType.FUZZY:
            # Simple fuzzy match - could use more sophisticated algorithm
            return self._fuzzy_match(text, pattern)
        
        return False
    
    def _fuzzy_match(self, text: str, pattern: str) -> bool:
        """Simple fuzzy string matching.
        
        Args:
            text: Text to check
            pattern: Pattern to match
            
        Returns:
            True if fuzzy match succeeds
        """
        # Simple implementation - checks if most characters are present
        if len(pattern) > len(text):
            return False
        
        matched = 0
        pattern_chars = list(pattern)
        for char in text:
            if pattern_chars and char == pattern_chars[0]:
                pattern_chars.pop(0)
                matched += 1
        
        return matched / len(pattern) >= self.confidence


class StateText:
    """Special state for text-based state identification.
    
    Port of StateText from Qontinui framework class.
    
    StateText provides text-centric state identification and validation.
    Unlike regular states that rely on images or regions, StateText identifies
    states based on text content, making it ideal for text-heavy applications,
    terminal interfaces, or dynamic content where visual elements change but
    text remains consistent.
    
    Key features:
    - Multiple text patterns for identification
    - Various matching strategies (exact, contains, regex)
    - Text extraction from regions
    - Validation based on text presence
    - Support for dynamic text with patterns
    
    Use cases:
    - Terminal/console applications
    - Text-based menus
    - Dynamic content with consistent text
    - Error message detection
    - Status message validation
    
    Example:
        # Create state that identifies login screen by text
        login_text_state = StateText("LoginScreen")
        login_text_state.add_pattern("Username:", TextMatchType.CONTAINS)
        login_text_state.add_pattern("Password:", TextMatchType.CONTAINS)
        login_text_state.add_pattern("Sign In", TextMatchType.EXACT)
        
        # Check if current screen matches
        if login_text_state.matches_current():
            print("On login screen")
    """
    
    def __init__(self, name: str, state_enum: Optional[StateEnum] = None):
        """Initialize StateText.
        
        Args:
            name: State name
            state_enum: Optional state enum
        """
        self._state = State(name=name, state_enum=state_enum)
        self._special_type = SpecialStateType.NULL  # Can be overridden
        self._text_patterns: List[TextPattern] = []
        self._required_texts: Set[str] = set()
        self._forbidden_texts: Set[str] = set()
        self._state_strings: List[StateString] = []
        self._text_regions: Dict[str, Any] = {}  # Region name -> Region
        self._extracted_text: str = ""
        self._match_all_patterns: bool = True  # All patterns must match
        
    @property
    def state(self) -> State:
        """Get underlying state."""
        return self._state
    
    @property
    def name(self) -> str:
        """Get state name."""
        return self._state.name
    
    @property
    def special_type(self) -> SpecialStateType:
        """Get special state type."""
        return self._special_type
    
    def set_special_type(self, special_type: SpecialStateType) -> 'StateText':
        """Set special state type.
        
        Args:
            special_type: Special state type
            
        Returns:
            Self for fluent interface
        """
        self._special_type = special_type
        return self
    
    def add_pattern(self, pattern: str, match_type: TextMatchType = TextMatchType.CONTAINS,
                   case_sensitive: bool = False, confidence: float = 0.8) -> 'StateText':
        """Add a text pattern for state identification.
        
        Args:
            pattern: Text pattern
            match_type: How to match the pattern
            case_sensitive: Whether to match case
            confidence: Confidence threshold for fuzzy matching
            
        Returns:
            Self for fluent interface
        """
        text_pattern = TextPattern(
            pattern=pattern,
            match_type=match_type,
            case_sensitive=case_sensitive,
            confidence=confidence
        )
        self._text_patterns.append(text_pattern)
        return self
    
    def add_regex_pattern(self, pattern: str, flags: int = 0) -> 'StateText':
        """Add a regex pattern for state identification.
        
        Args:
            pattern: Regular expression pattern
            flags: Regex flags (re.IGNORECASE, etc.)
            
        Returns:
            Self for fluent interface
        """
        text_pattern = TextPattern(
            pattern=pattern,
            match_type=TextMatchType.REGEX,
            regex_flags=flags
        )
        self._text_patterns.append(text_pattern)
        return self
    
    def add_required_text(self, text: str) -> 'StateText':
        """Add text that must be present.
        
        Args:
            text: Required text
            
        Returns:
            Self for fluent interface
        """
        self._required_texts.add(text)
        return self
    
    def add_forbidden_text(self, text: str) -> 'StateText':
        """Add text that must not be present.
        
        Args:
            text: Forbidden text
            
        Returns:
            Self for fluent interface
        """
        self._forbidden_texts.add(text)
        return self
    
    def add_state_string(self, state_string: StateString) -> 'StateText':
        """Add a StateString for text extraction.
        
        Args:
            state_string: StateString to add
            
        Returns:
            Self for fluent interface
        """
        self._state_strings.append(state_string)
        return self
    
    def set_match_all(self, match_all: bool) -> 'StateText':
        """Set whether all patterns must match.
        
        Args:
            match_all: If True, all patterns must match. If False, any pattern matching is sufficient.
            
        Returns:
            Self for fluent interface
        """
        self._match_all_patterns = match_all
        return self
    
    def matches_text(self, text: str) -> bool:
        """Check if text matches this state's patterns.
        
        Args:
            text: Text to check
            
        Returns:
            True if matches state patterns
        """
        # Check forbidden texts first
        text_lower = text.lower()
        for forbidden in self._forbidden_texts:
            if forbidden.lower() in text_lower:
                return False
        
        # Check required texts
        for required in self._required_texts:
            if required.lower() not in text_lower:
                return False
        
        # Check patterns
        if not self._text_patterns:
            return True  # No patterns means only required/forbidden matter
        
        matches = [pattern.matches(text) for pattern in self._text_patterns]
        
        if self._match_all_patterns:
            return all(matches)
        else:
            return any(matches)
    
    def extract_text(self) -> str:
        """Extract text from configured regions and state strings.
        
        Returns:
            Extracted text
        """
        extracted = []
        
        # Extract from StateStrings
        for state_string in self._state_strings:
            if hasattr(state_string, 'get_text'):
                text = state_string.get_text()
                if text:
                    extracted.append(text)
        
        # Would extract from regions here if OCR was available
        # for region_name, region in self._text_regions.items():
        #     text = ocr.extract_text(region)
        #     if text:
        #         extracted.append(text)
        
        self._extracted_text = '\n'.join(extracted)
        return self._extracted_text
    
    def matches_current(self) -> bool:
        """Check if current screen matches this text state.
        
        Returns:
            True if current screen matches
        """
        current_text = self.extract_text()
        return self.matches_text(current_text)
    
    def get_patterns(self) -> List[TextPattern]:
        """Get all text patterns.
        
        Returns:
            List of text patterns
        """
        return self._text_patterns.copy()
    
    def get_required_texts(self) -> Set[str]:
        """Get required texts.
        
        Returns:
            Set of required texts
        """
        return self._required_texts.copy()
    
    def get_forbidden_texts(self) -> Set[str]:
        """Get forbidden texts.
        
        Returns:
            Set of forbidden texts
        """
        return self._forbidden_texts.copy()
    
    def clear_patterns(self) -> 'StateText':
        """Clear all patterns.
        
        Returns:
            Self for fluent interface
        """
        self._text_patterns.clear()
        return self
    
    def clear_required(self) -> 'StateText':
        """Clear required texts.
        
        Returns:
            Self for fluent interface
        """
        self._required_texts.clear()
        return self
    
    def clear_forbidden(self) -> 'StateText':
        """Clear forbidden texts.
        
        Returns:
            Self for fluent interface
        """
        self._forbidden_texts.clear()
        return self
    
    def is_valid(self) -> bool:
        """Check if state configuration is valid.
        
        Returns:
            True if valid
        """
        # Must have at least one way to identify state
        return (bool(self._text_patterns) or 
                bool(self._required_texts) or
                bool(self._state_strings))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            'name': self._state.name,
            'special_type': self._special_type.name,
            'patterns': [
                {
                    'pattern': p.pattern,
                    'match_type': p.match_type.name,
                    'case_sensitive': p.case_sensitive,
                    'confidence': p.confidence
                }
                for p in self._text_patterns
            ],
            'required_texts': list(self._required_texts),
            'forbidden_texts': list(self._forbidden_texts),
            'match_all': self._match_all_patterns
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateText':
        """Create from dictionary representation.
        
        Args:
            data: Dictionary data
            
        Returns:
            New StateText instance
        """
        state_text = cls(data['name'])
        
        if 'special_type' in data:
            state_text.set_special_type(SpecialStateType[data['special_type']])
        
        for pattern_data in data.get('patterns', []):
            state_text.add_pattern(
                pattern_data['pattern'],
                TextMatchType[pattern_data['match_type']],
                pattern_data.get('case_sensitive', False),
                pattern_data.get('confidence', 0.8)
            )
        
        for text in data.get('required_texts', []):
            state_text.add_required_text(text)
        
        for text in data.get('forbidden_texts', []):
            state_text.add_forbidden_text(text)
        
        state_text.set_match_all(data.get('match_all', True))
        
        return state_text
    
    def __str__(self) -> str:
        """String representation."""
        return f"StateText({self._state.name}, {len(self._text_patterns)} patterns)"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return (f"StateText(name='{self._state.name}', "
                f"patterns={len(self._text_patterns)}, "
                f"required={len(self._required_texts)}, "
                f"forbidden={len(self._forbidden_texts)})")