"""Model Element package - ported from Qontinui framework's model/element.

Core element types used throughout the framework for representing
locations, regions, images, patterns, and other UI elements.
"""

from .anchor import Anchor, AnchorType
from .anchors import Anchors
from .color import HSV, RGB
from .grid import Grid, GridBuilder
from .image import Image
from .location import Location
from .match_adjustment_options import MatchAdjustmentOptions
from .movement import Movement
from .overlapping_grids import OverlappingGrids
from .pattern import Pattern
from .position import Position
from .positions import PositionName, Positions
from .region import Region
from .region_factory import RegionFactory
from .region_geometry import RegionGeometry
from .region_transforms import RegionTransforms
from .scene import Scene
from .search_region_on_object import SearchRegionOnObject, StateObjectType
from .text import Text

__all__ = [
    # Spatial types
    "Region",
    "RegionFactory",
    "RegionGeometry",
    "RegionTransforms",
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
    "SearchRegionOnObject",
    "StateObjectType",
    "MatchAdjustmentOptions",
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
