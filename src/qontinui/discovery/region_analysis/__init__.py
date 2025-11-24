"""Region analysis module for analyzing and classifying screen regions.

This module provides functionality for dividing screenshots into meaningful regions,
analyzing their properties, and understanding spatial relationships between different
areas of the screen.

Region analysis is critical for understanding the structure of UI layouts and
identifying logical groupings of elements that may represent different states
or functional areas of an application.

Key Components:
    - Screen segmentation and region extraction
    - Region property analysis (size, position, content type)
    - Spatial relationship detection
    - Region clustering and grouping
    - Dynamic region tracking across frames

Example:
    >>> from qontinui.discovery.region_analysis import analyze_regions
    >>> regions = analyze_regions(screenshot)
    >>> for region in regions:
    ...     print(f"Region at {region.bounds} contains {region.element_count} elements")
"""

# Pattern detectors
from .grid_pattern_detector import GridPatternDetector

# Window component detectors
from .window_border_detector import WindowBorderDetector
from .window_title_bar_detector import WindowTitleBarDetector
from .window_close_button_detector import WindowCloseButtonDetector

# Slot and texture detectors
from .slot_border_detector import SlotBorderDetector
from .texture_uniformity_detector import TextureUniformityDetector

# Text detection variants
from .connected_components_text_detector import ConnectedComponentsTextDetector
from .contour_text_detector import ContourTextDetector
from .edge_morphology_text_detector import EdgeMorphologyTextDetector
from .gradient_text_detector import GradientTextDetector
from .mser_text_detector import MSERTextDetector
from .ocr_text_detector import OCRTextDetector
from .stroke_width_text_detector import StrokeWidthTextDetector

# Grid detection variants
from .contour_grid_detector import ContourGridDetector
from .hough_grid_detector import HoughGridDetector
from .ransac_grid_detector import RANSACGridDetector
from .template_grid_detector import TemplateGridDetector

# Other detectors
from .corner_clustering_detector import CornerClusteringDetector
from .color_quantization_detector import ColorQuantizationDetector
from .frequency_analysis_detector import FrequencyAnalysisDetector

__all__ = [
    # Pattern detectors
    "GridPatternDetector",

    # Window component detectors
    "WindowBorderDetector",
    "WindowTitleBarDetector",
    "WindowCloseButtonDetector",

    # Slot and texture detectors
    "SlotBorderDetector",
    "TextureUniformityDetector",

    # Text detection variants
    "ConnectedComponentsTextDetector",
    "ContourTextDetector",
    "EdgeMorphologyTextDetector",
    "GradientTextDetector",
    "MSERTextDetector",
    "OCRTextDetector",
    "StrokeWidthTextDetector",

    # Grid detection variants
    "ContourGridDetector",
    "HoughGridDetector",
    "RANSACGridDetector",
    "TemplateGridDetector",

    # Other detectors
    "CornerClusteringDetector",
    "ColorQuantizationDetector",
    "FrequencyAnalysisDetector",
]

__version__ = "0.1.0"
