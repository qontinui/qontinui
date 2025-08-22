"""Match class - ported from Qontinui framework.

Wrapper around MatchObject with additional functionality.
"""

from dataclasses import dataclass
from typing import Optional, Any
from ..model.element import Location, Region, Pattern
from ..model.match import MatchObject
from ..actions import ActionResult


@dataclass
class Match:
    """Enhanced match result with action capabilities.
    
    Port of Match from Qontinui framework class.
    Extends MatchObject with action methods.
    """
    
    match_object: MatchObject
    
    def __post_init__(self):
        """Initialize match properties."""
        # Expose key properties directly
        self.location = self.match_object.location
        self.region = self.match_object.region
        self.similarity = self.match_object.similarity
        self.pattern = self.match_object.pattern
    
    @property
    def center(self) -> Location:
        """Get center location of match."""
        return self.match_object.center
    
    @property
    def target(self) -> Location:
        """Get target location for actions."""
        return self.match_object.target
    
    @property
    def score(self) -> float:
        """Get similarity score."""
        return self.match_object.similarity
    
    def exists(self) -> bool:
        """Check if match exists (was found).
        
        Returns:
            True if match exists
        """
        return self.match_object is not None and self.similarity > 0
    
    def click(self) -> ActionResult:
        """Click on this match.
        
        Returns:
            ActionResult from click action
        """
        if not self.exists():
            return ActionResult(success=False, error="Match does not exist")
        
        # This would use the Action system to click
        from ..actions import Action, ClickOptions
        action = Action(ClickOptions())
        return action.click(self.target.x, self.target.y)
    
    def double_click(self) -> ActionResult:
        """Double-click on this match.
        
        Returns:
            ActionResult from double-click action
        """
        if not self.exists():
            return ActionResult(success=False, error="Match does not exist")
        
        from ..actions import Action, ClickOptions
        action = Action(ClickOptions().double_click())
        return action.click(self.target.x, self.target.y)
    
    def right_click(self) -> ActionResult:
        """Right-click on this match.
        
        Returns:
            ActionResult from right-click action
        """
        if not self.exists():
            return ActionResult(success=False, error="Match does not exist")
        
        from ..actions import Action, ClickOptions
        action = Action(ClickOptions().right_click())
        return action.click(self.target.x, self.target.y)
    
    def hover(self) -> ActionResult:
        """Move mouse to this match.
        
        Returns:
            ActionResult from move action
        """
        if not self.exists():
            return ActionResult(success=False, error="Match does not exist")
        
        from ..actions import Action
        action = Action()
        return action.move(self.target.x, self.target.y)
    
    def drag_to(self, target: 'Match') -> ActionResult:
        """Drag from this match to another.
        
        Args:
            target: Target match to drag to
            
        Returns:
            ActionResult from drag action
        """
        if not self.exists():
            return ActionResult(success=False, error="Source match does not exist")
        if not target.exists():
            return ActionResult(success=False, error="Target match does not exist")
        
        from ..actions import Action, DragOptions
        action = Action(DragOptions())
        return action.drag(
            self.target.x, self.target.y,
            target.target.x, target.target.y
        )
    
    def type_text(self, text: str) -> ActionResult:
        """Type text at this match location.
        
        Args:
            text: Text to type
            
        Returns:
            ActionResult from type action
        """
        if not self.exists():
            return ActionResult(success=False, error="Match does not exist")
        
        # Click first to focus
        click_result = self.click()
        if not click_result.success:
            return click_result
        
        from ..actions import Action, TypeOptions
        action = Action(TypeOptions())
        return action.type_text(text)
    
    def highlight(self, duration: float = 2.0) -> ActionResult:
        """Highlight this match on screen.
        
        Args:
            duration: How long to show highlight
            
        Returns:
            ActionResult from highlight action
        """
        if not self.exists():
            return ActionResult(success=False, error="Match does not exist")
        
        # This would draw a rectangle around the match
        # Implementation depends on platform
        return ActionResult(
            success=True,
            data={'region': self.region, 'duration': duration}
        )
    
    def get_text(self) -> str:
        """Extract text from this match region using OCR.
        
        Returns:
            Extracted text or empty string
        """
        if not self.exists():
            return ""
        
        # This would use OCR on the match region
        # For now, return placeholder
        return f"Text from {self.region}"
    
    def distance_to(self, other: 'Match') -> float:
        """Calculate distance to another match.
        
        Args:
            other: Other match
            
        Returns:
            Distance in pixels or float('inf') if either doesn't exist
        """
        if not self.exists() or not other.exists():
            return float('inf')
        
        return self.center.distance_to(other.center)
    
    def is_left_of(self, other: 'Match') -> bool:
        """Check if this match is to the left of another.
        
        Args:
            other: Other match
            
        Returns:
            True if this match is to the left
        """
        if not self.exists() or not other.exists():
            return False
        
        return self.region.right <= other.region.left
    
    def is_right_of(self, other: 'Match') -> bool:
        """Check if this match is to the right of another.
        
        Args:
            other: Other match
            
        Returns:
            True if this match is to the right
        """
        if not self.exists() or not other.exists():
            return False
        
        return self.region.left >= other.region.right
    
    def is_above(self, other: 'Match') -> bool:
        """Check if this match is above another.
        
        Args:
            other: Other match
            
        Returns:
            True if this match is above
        """
        if not self.exists() or not other.exists():
            return False
        
        return self.region.bottom <= other.region.top
    
    def is_below(self, other: 'Match') -> bool:
        """Check if this match is below another.
        
        Args:
            other: Other match
            
        Returns:
            True if this match is below
        """
        if not self.exists() or not other.exists():
            return False
        
        return self.region.top >= other.region.bottom
    
    def __str__(self) -> str:
        """String representation."""
        if not self.exists():
            return "Match(not found)"
        return str(self.match_object)
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"Match({self.match_object!r})"
    
    def __bool__(self) -> bool:
        """Boolean evaluation - True if match exists."""
        return self.exists()
    
    @classmethod
    def empty(cls) -> 'Match':
        """Create an empty (not found) match.
        
        Returns:
            Empty Match instance
        """
        return cls(MatchObject(
            location=Location(0, 0),
            region=Region(0, 0, 0, 0),
            similarity=0.0
        ))