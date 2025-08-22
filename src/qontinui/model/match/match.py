"""Match model - ported from Qontinui framework.

Represents a successful pattern match found on the screen.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from ..element.location import Location
from ..element.region import Region
from ..element.image import Image
from ..element.position import Position


@dataclass
class Match:
    """Represents a successful pattern match found on the screen.
    
    Port of Match from Qontinui framework class.
    
    A Match is created when a Find operation successfully locates a GUI element (image, text, or region) 
    on the screen. It encapsulates all information about the match including its location, similarity score, 
    the image content at that location, and metadata about how it was found.
    
    In the model-based approach, matches are fundamental for:
    - Providing targets for mouse and keyboard actions (clicks, typing, etc.)
    - Verifying that GUI elements exist in expected States
    - Building dynamic relationships between GUI elements
    - Creating visual feedback through match highlighting
    - Tracking interaction history with GUI elements
    
    Key features:
    - Score: Similarity score (0.0-1.0) indicating match quality
    - Target: Location within the matched region for precise interactions
    - Image content: Actual pixels from the screen at the match location
    - Search image: The pattern that was searched for
    - State context: Information about which State object found this match
    
    Unlike MatchSnapshot which can represent failed matches, a Match object always 
    represents a successful find operation. Multiple Match objects are aggregated in 
    an ActionResult.
    """
    
    score: float = 0.0
    """Similarity score (0.0-1.0) indicating match quality."""
    
    target: Optional[Location] = None
    """Location within the matched region for precise interactions."""
    
    image: Optional[Image] = None
    """Actual pixels from the screen at the match location."""
    
    text: str = ""
    """Text content if this is a text match."""
    
    name: str = ""
    """Name identifier for this match."""
    
    search_image: Optional[Image] = None
    """The image used to find the match."""
    
    anchors: Optional['Anchors'] = None
    """Anchors associated with this match."""
    
    state_object_data: Optional['StateObjectMetadata'] = None
    """Metadata about the State object that found this match."""
    
    histogram: Optional[np.ndarray] = field(default=None, repr=False)
    """Histogram data for this match."""
    
    scene: Optional['Scene'] = None
    """Scene containing this match."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    """When this match was found."""
    
    times_acted_on: int = 0
    """Number of times actions have been performed on this match."""
    
    def __post_init__(self):
        """Initialize match with region if not set."""
        if self.target is None:
            self.target = Location(region=Region())
    
    @classmethod
    def from_region(cls, region: Region) -> 'Match':
        """Create Match from Region.
        
        Args:
            region: Region to create match from
            
        Returns:
            Match instance
        """
        return cls(target=Location(region=region))
    
    @property
    def x(self) -> int:
        """Get x coordinate.
        
        Returns:
            X coordinate or 0
        """
        region = self.get_region()
        return region.x if region else 0
    
    @property
    def y(self) -> int:
        """Get y coordinate.
        
        Returns:
            Y coordinate or 0
        """
        region = self.get_region()
        return region.y if region else 0
    
    @property
    def w(self) -> int:
        """Get width.
        
        Returns:
            Width or 0
        """
        region = self.get_region()
        return region.width if region else 0
    
    @property
    def h(self) -> int:
        """Get height.
        
        Returns:
            Height or 0
        """
        region = self.get_region()
        return region.height if region else 0
    
    def get_region(self) -> Optional[Region]:
        """Get the region of this match.
        
        Returns:
            Region or None
        """
        if self.target is None:
            return None
        return self.target.region
    
    def set_region(self, region: Region) -> None:
        """Set the region of this match.
        
        Args:
            region: New region
        """
        if self.target is None:
            self.target = Location(region=region)
        else:
            self.target.region = region
    
    def get_mat(self) -> Optional[np.ndarray]:
        """Get the image as BGR NumPy array.
        
        Returns:
            BGR array or None
        """
        return self.image.get_mat_bgr() if self.image else None
    
    def compare_by_score(self, other: 'Match') -> float:
        """Compare this match to another by score.
        
        Args:
            other: Other match
            
        Returns:
            Score difference
        """
        return self.score - other.score
    
    def size(self) -> int:
        """Get area of the match region.
        
        Returns:
            Area in pixels
        """
        region = self.get_region()
        return region.area if region else 0
    
    def increment_times_acted_on(self) -> None:
        """Increment the times acted on counter."""
        self.times_acted_on += 1
    
    def set_image_with_scene(self) -> None:
        """Set image from scene if available."""
        if self.scene is None:
            return
        
        # Extract sub-image from scene
        # This would need implementation of BufferedImageUtilities
        # For now, just a placeholder
        pass
    
    def get_owner_state_name(self) -> str:
        """Get the name of the owner state.
        
        Returns:
            Owner state name or empty string
        """
        if self.state_object_data:
            return self.state_object_data.owner_state_name
        return ""
    
    def to_state_image(self) -> 'StateImage':
        """Convert this match to a StateImage.
        
        If there is a StateObject, we try to recreate it as a StateImage.
        
        Returns:
            StateImage created from this match
        """
        from ..state.state_image import StateImage
        from ..element.pattern import Pattern
        
        state_image = StateImage()
        state_image.name = self.name
        
        if self.state_object_data:
            state_image.owner_state_name = self.state_object_data.owner_state_name
            if self.state_object_data.state_object_name:
                state_image.name = self.state_object_data.state_object_name
        
        state_image.add_patterns(Pattern.from_match(self))
        return state_image
    
    def __str__(self) -> str:
        """String representation."""
        parts = ["M["]
        
        if self.name:
            parts.append(f"#{self.name}# ")
        
        region = self.get_region()
        if region:
            parts.append(f"R[{region.x},{region.y} {region.width}x{region.height}] simScore:{self.score:.1f}")
        else:
            parts.append(f"R[null] simScore:{self.score:.1f}")
        
        if self.text:
            parts.append(f" text:{self.text}")
        
        parts.append("]")
        return "".join(parts)
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()
    
    def __eq__(self, other) -> bool:
        """Check equality."""
        if not isinstance(other, Match):
            return False
        
        return (self.score == other.score and
                self.name == other.name and
                self.text == other.text and
                self.get_region() == other.get_region())
    
    def __hash__(self) -> int:
        """Get hash code."""
        return hash((self.score, self.name, self.text, self.get_region()))


class MatchBuilder:
    """Builder for creating Match objects.
    
    Port of Match from Qontinui framework.Builder class.
    """
    
    def __init__(self):
        """Initialize builder with defaults."""
        self.target = Location()
        self.position = Position()
        self.offset_x = 0
        self.offset_y = 0
        self.image = None
        self.search_image = None
        self.region = None
        self.name = None
        self.text = None
        self.anchors = None
        self.state_object_data = None
        self.histogram = None
        self.scene = None
        self.sim_score = -1
    
    def set_match(self, match: Match) -> 'MatchBuilder':
        """Copy from existing match.
        
        Args:
            match: Match to copy from
            
        Returns:
            Self for chaining
        """
        if match.image:
            self.image = match.image
        if match.search_image:
            self.search_image = match.search_image
        if match.get_region():
            self.set_region(match.get_region())
        if match.name:
            self.name = match.name
        if match.text:
            self.text = match.text
        if match.anchors:
            self.anchors = match.anchors
        if match.state_object_data:
            self.state_object_data = match.state_object_data
        if match.histogram is not None:
            self.histogram = match.histogram
        if match.scene:
            self.scene = match.scene
        self.sim_score = match.score
        return self
    
    def set_region(self, region: Region) -> 'MatchBuilder':
        """Set region.
        
        Args:
            region: Region to set
            
        Returns:
            Self for chaining
        """
        self.region = region
        return self
    
    def set_position(self, position: Position) -> 'MatchBuilder':
        """Set position.
        
        Args:
            position: Position to set
            
        Returns:
            Self for chaining
        """
        self.position = position
        return self
    
    def set_offset_x(self, offset_x: int) -> 'MatchBuilder':
        """Set x offset.
        
        Args:
            offset_x: X offset
            
        Returns:
            Self for chaining
        """
        self.offset_x = offset_x
        return self
    
    def set_offset_y(self, offset_y: int) -> 'MatchBuilder':
        """Set y offset.
        
        Args:
            offset_y: Y offset
            
        Returns:
            Self for chaining
        """
        self.offset_y = offset_y
        return self
    
    def set_offset(self, offset: Location) -> 'MatchBuilder':
        """Set offset from location.
        
        Args:
            offset: Offset location
            
        Returns:
            Self for chaining
        """
        self.offset_x = offset.get_final_location().x
        self.offset_y = offset.get_final_location().y
        return self
    
    def set_image(self, image: Image) -> 'MatchBuilder':
        """Set image.
        
        Args:
            image: Image to set
            
        Returns:
            Self for chaining
        """
        self.image = image
        return self
    
    def set_region_xywh(self, x: int, y: int, w: int, h: int) -> 'MatchBuilder':
        """Set region from coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            w: Width
            h: Height
            
        Returns:
            Self for chaining
        """
        self.region = Region(x=x, y=y, width=w, height=h)
        return self
    
    def set_name(self, name: str) -> 'MatchBuilder':
        """Set name.
        
        Args:
            name: Name to set
            
        Returns:
            Self for chaining
        """
        self.name = name
        return self
    
    def set_text(self, text: str) -> 'MatchBuilder':
        """Set text.
        
        Args:
            text: Text to set
            
        Returns:
            Self for chaining
        """
        self.text = text
        return self
    
    def set_search_image(self, image: Image) -> 'MatchBuilder':
        """Set search image.
        
        Args:
            image: Search image
            
        Returns:
            Self for chaining
        """
        self.search_image = image
        return self
    
    def set_anchors(self, anchors: 'Anchors') -> 'MatchBuilder':
        """Set anchors.
        
        Args:
            anchors: Anchors to set
            
        Returns:
            Self for chaining
        """
        self.anchors = anchors
        return self
    
    def set_state_object_data(self, state_object_data: 'StateObjectMetadata') -> 'MatchBuilder':
        """Set state object data.
        
        Args:
            state_object_data: State object metadata
            
        Returns:
            Self for chaining
        """
        self.state_object_data = state_object_data
        return self
    
    def set_histogram(self, histogram: np.ndarray) -> 'MatchBuilder':
        """Set histogram.
        
        Args:
            histogram: Histogram array
            
        Returns:
            Self for chaining
        """
        self.histogram = histogram
        return self
    
    def set_scene(self, scene: 'Scene') -> 'MatchBuilder':
        """Set scene.
        
        Args:
            scene: Scene to set
            
        Returns:
            Self for chaining
        """
        self.scene = scene
        return self
    
    def set_sim_score(self, sim_score: float) -> 'MatchBuilder':
        """Set similarity score.
        
        Args:
            sim_score: Similarity score
            
        Returns:
            Self for chaining
        """
        self.sim_score = sim_score
        return self
    
    def build(self) -> Match:
        """Build the Match object.
        
        Returns:
            Constructed Match
        """
        match = Match(target=Location(region=Region()))
        
        # Set target location
        if self.region:
            self.target.region = self.region
        self.target.position = self.position
        self.target.offset_x = self.offset_x
        self.target.offset_y = self.offset_y
        match.target = self.target
        
        # Set other fields
        match.scene = self.scene
        
        # Set match image
        if self.image:
            match.image = self.image
        elif match.scene:
            match.set_image_with_scene()
        
        if self.name:
            match.name = self.name
        if self.text:
            match.text = self.text
        if self.sim_score >= 0:
            match.score = self.sim_score
        
        match.anchors = self.anchors
        match.state_object_data = self.state_object_data
        match.histogram = self.histogram
        match.timestamp = datetime.now()
        match.search_image = self.search_image
        
        return match