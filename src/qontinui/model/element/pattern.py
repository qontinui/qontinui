"""Pattern model - ported from Qontinui framework.

Represents a visual template for pattern matching in the GUI automation framework.
"""

from typing import Optional
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path

from .image import Image
from .position import Position
from .location import Location
from .region import Region
from ..match.match import Match


@dataclass
class Pattern:
    """Represents a visual template for pattern matching.
    
    Port of Pattern from Qontinui framework class.
    
    Pattern is the fundamental unit of visual recognition in Qontinui, encapsulating an image 
    template along with its matching parameters and metadata. It serves as the building block 
    for StateImages and provides the core pattern matching capability that enables visual 
    GUI automation.
    
    Key components:
    - Image Data: The actual visual template stored as an Image
    - Search Configuration: Regions where this pattern should be searched for
    - Target Parameters: Position and offset for precise interaction points
    - Match History: Historical data for mocking and analysis
    - Anchors: Reference points for defining relative positions
    
    Pattern types:
    - Fixed: Patterns that always appear in the same screen location
    - Dynamic: Patterns with changing content that require special handling
    - Standard: Regular patterns that can appear anywhere within search regions
    
    In the model-based approach, Patterns provide the visual vocabulary for describing 
    GUI elements. They enable the framework to recognize and interact with GUI components 
    regardless of the underlying technology, making automation truly cross-platform and 
    technology-agnostic.
    """
    
    url: Optional[str] = None
    """URL source of the pattern."""
    
    imgpath: Optional[str] = None
    """File path to the image."""
    
    name: str = ""
    """Name of this pattern."""
    
    fixed: bool = False
    """Whether this pattern always appears in the same location."""
    
    search_regions: 'SearchRegions' = field(default_factory=lambda: SearchRegions())
    """Regions where this pattern should be searched for."""
    
    set_kmeans_color_profiles: bool = False
    """Whether to compute expensive color profiles."""
    
    match_history: 'ActionHistory' = field(default_factory=lambda: ActionHistory())
    """History of matches for this pattern."""
    
    index: int = 0
    """Unique identifier for classification matrices."""
    
    dynamic: bool = False
    """Whether this is a dynamic image that cannot be found using pattern matching."""
    
    target_position: Position = field(default_factory=lambda: Position(0.5, 0.5))
    """Position within the pattern for converting to a location."""
    
    target_offset: Location = field(default_factory=lambda: Location(0, 0))
    """Offset to adjust the match location."""
    
    anchors: 'Anchors' = field(default_factory=lambda: Anchors())
    """Reference points for defining relative positions."""
    
    image: Optional[Image] = None
    """The actual image data."""
    
    @classmethod
    def from_file(cls, img_path: str) -> 'Pattern':
        """Create Pattern from image file.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Pattern instance
        """
        if not img_path:
            return cls()
        
        pattern = cls(imgpath=img_path)
        pattern._set_name_from_filename_if_empty(img_path)
        
        # Load image
        pattern.image = Image.from_file(img_path)
        if pattern.image and not pattern.name:
            pattern.name = pattern.image.name
        
        return pattern
    
    @classmethod
    def from_image(cls, image: Image) -> 'Pattern':
        """Create Pattern from Image.
        
        Args:
            image: Image object
            
        Returns:
            Pattern instance
        """
        return cls(image=image, name=image.name if image else "")
    
    @classmethod
    def from_match(cls, match: Match) -> 'Pattern':
        """Create Pattern from Match.
        
        Args:
            match: Match object
            
        Returns:
            Pattern instance
        """
        pattern = cls()
        
        # Use match image or search image
        image_to_use = match.image if match.image else match.search_image
        
        pattern.fixed = True
        pattern.search_regions.set_fixed_region(match.get_region())
        
        if image_to_use:
            pattern.image = image_to_use
            pattern.name = match.name
        else:
            pattern.imgpath = match.name
        
        return pattern
    
    @classmethod
    def from_numpy(cls, mat: np.ndarray, name: Optional[str] = None) -> 'Pattern':
        """Create Pattern from NumPy array.
        
        Args:
            mat: NumPy array (BGR format)
            name: Optional name
            
        Returns:
            Pattern instance
        """
        return cls(image=Image.from_numpy(mat, name=name), name=name or "")
    
    def _set_name_from_filename_if_empty(self, filename: str) -> None:
        """Set name from filename if not already set.
        
        Args:
            filename: File path
        """
        if not self.name and filename:
            path = Path(filename)
            self.name = path.stem  # Filename without extension
    
    def get_b_image(self) -> Optional[np.ndarray]:
        """Get the image as BGR NumPy array.
        
        Returns:
            BGR array or None
        """
        return self.image.get_mat_bgr() if self.image else None
    
    def is_defined(self) -> bool:
        """Check if pattern has valid image data.
        
        Returns:
            True if pattern has image
        """
        return self.image is not None and not self.image.is_empty()
    
    def __str__(self) -> str:
        """String representation."""
        parts = [f"Pattern('{self.name}'"]
        if self.fixed:
            parts.append(" fixed")
        if self.dynamic:
            parts.append(" dynamic")
        if self.image:
            parts.append(f" {self.image.w}x{self.image.h}")
        parts.append(")")
        return "".join(parts)


class SearchRegions:
    """Placeholder for SearchRegions class.
    
    Will be implemented when migrating the search regions functionality.
    """
    
    def __init__(self):
        """Initialize empty search regions."""
        self.fixed_region = None
        self.regions = []
    
    def set_fixed_region(self, region: Optional[Region]) -> None:
        """Set the fixed region.
        
        Args:
            region: Fixed region or None
        """
        self.fixed_region = region
    
    def get_fixed_region(self) -> Optional[Region]:
        """Get the fixed region.
        
        Returns:
            Fixed region or None
        """
        return self.fixed_region
    
    def is_defined(self) -> bool:
        """Check if any regions are defined.
        
        Returns:
            True if regions are defined
        """
        return self.fixed_region is not None or bool(self.regions)


class Anchors:
    """Placeholder for Anchors class.
    
    Will be implemented when migrating the anchors functionality.
    """
    pass


class ActionHistory:
    """Placeholder for ActionHistory class.
    
    Will be implemented when migrating the action history functionality.
    """
    pass