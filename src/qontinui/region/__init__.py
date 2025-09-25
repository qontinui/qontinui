"""Declarative region definition package for Qontinui.

Provides fluent and declarative APIs for defining regions,
following Brobot's approach to spatial relationships.
"""

from .anchor import Anchor, AnchorPoint
from .region_builder import RegionBuilder
from .region_definitions import RegionDefinitions
from .region_offset import RegionOffset
from .relative_region import RelativeRegion

__all__ = [
    "RegionBuilder",
    "Anchor",
    "AnchorPoint",
    "RegionOffset",
    "RegionDefinitions",
    "RelativeRegion",
]
