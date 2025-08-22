"""Model Element package - ported from Qontinui framework's model/element.

Core element types used throughout the framework for representing
locations, regions, images, patterns, and other UI elements.
"""

from .region import Region
from .location import Location
from .position import Position
from .anchor import Anchor, AnchorType
from .anchors import Anchors
from .image import Image
from .pattern import Pattern
from .color import RGB, HSV
from .grid import Grid, GridBuilder
from .scene import Scene
from .text import Text
from .positions import Positions, PositionName
from .movement import Movement
from .overlapping_grids import OverlappingGrids

__all__ = [
    # Spatial types
    "Region",
    "Location",
    "Position",
    "Anchor",
    "AnchorType",
    "Grid",
    "GridBuilder",
    "OverlappingGrids",
    "Anchors",
    "Positions",
    "PositionName",
    "Movement",
    
    # Image types
    "Image",
    "Pattern",
    "Scene",
    
    # Text types
    "Text",
    
    # Color types
    "RGB",
    "HSV",
]