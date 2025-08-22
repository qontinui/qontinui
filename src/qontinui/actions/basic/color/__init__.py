"""Color-based actions and utilities - ported from Qontinui framework.

Provides color-based pattern matching, classification, and analysis.
"""

from .classify import Classify
from .find_color import FindColor, ColorProfile
from .color_find_options import (
    ColorFindOptions,
    ColorStrategy,
    AreaFilteringOptions,
    HSVBinOptions
)
from .color_cluster import ColorCluster, ColorClusterAnalyzer
from .color_statistics import ColorStatistics, ColorStatisticsAnalyzer

__all__ = [
    # Main actions
    'Classify',
    'FindColor',
    
    # Options and configuration
    'ColorFindOptions',
    'ColorStrategy',
    'AreaFilteringOptions',
    'HSVBinOptions',
    
    # Analysis utilities
    'ColorProfile',
    'ColorCluster',
    'ColorClusterAnalyzer',
    'ColorStatistics',
    'ColorStatisticsAnalyzer',
]