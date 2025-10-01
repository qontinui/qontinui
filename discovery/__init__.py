"""
State Discovery Module for Qontinui
Automated UI state detection and analysis
"""

from .deletion_manager import DeletionManager
from .models import (
    AnalysisConfig,
    AnalysisResult,
    DeleteOptions,
    DeleteResult,
    DeletionImpact,
    DiscoveredState,
    StateImage,
)
from .pixel_stability_analyzer import PixelStabilityAnalyzer
from .pixel_stability_matrix_analyzer import PixelStabilityMatrixAnalyzer

# For backward compatibility
StableRegionExtractor = PixelStabilityAnalyzer

__all__ = [
    "PixelStabilityAnalyzer",
    "PixelStabilityMatrixAnalyzer",
    "StableRegionExtractor",
    "StateImage",
    "DiscoveredState",
    "AnalysisResult",
    "AnalysisConfig",
    "DeleteOptions",
    "DeletionImpact",
    "DeleteResult",
    "DeletionManager",
]
