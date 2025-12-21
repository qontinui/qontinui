"""Color-based actions and utilities - ported from Qontinui framework.

Provides color-based pattern matching, classification, and analysis.
"""

from .classify import Classify
from .color_cluster import ColorCluster, ColorClusterAnalyzer
from .color_find_options import (
    AreaFilteringOptions,
    AreaFilteringOptionsBuilder,
    ColorFindOptions,
    ColorFindOptionsBuilder,
    ColorStrategy,
    HSVBinOptions,
    HSVBinOptionsBuilder,
)
from .color_statistics import ColorStatistics, ColorStatisticsAnalyzer
from .color_profile import ColorProfile
from .find_color import FindColor

__all__ = [
    # Main actions
    "Classify",
    "FindColor",
    # Options and configuration
    "ColorFindOptions",
    "ColorFindOptionsBuilder",
    "ColorStrategy",
    "AreaFilteringOptions",
    "AreaFilteringOptionsBuilder",
    "HSVBinOptions",
    "HSVBinOptionsBuilder",
    # Analysis utilities
    "ColorProfile",
    "ColorCluster",
    "ColorClusterAnalyzer",
    "ColorStatistics",
    "ColorStatisticsAnalyzer",
]
