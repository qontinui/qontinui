"""Element detection module for identifying UI elements in screenshots.

This module provides functionality for detecting and classifying UI elements such as
buttons, text fields, icons, and other interactive components using various computer
vision and machine learning techniques.

The element detection system forms the foundation of the discovery pipeline, enabling
automated identification of UI components that can be used for state construction
and application navigation.

Key Components:
    - Base classes for building analyzers (AnalysisType, BaseAnalyzer, etc.)
    - Button detectors (color, shape, shadow, hover, ensemble, fusion)
    - Input field detector
    - Icon button detector
    - Dropdown detector
    - Modal dialog detector
    - Sidebar detector
    - Menu bar detector
    - Typography detector

Example:
    >>> from qontinui.discovery.element_detection import InputFieldDetector
    >>> from qontinui.discovery.element_detection import AnalysisInput
    >>> detector = InputFieldDetector()
    >>> result = await detector.analyze(input_data)
    >>> for element in result.elements:
    ...     print(f"Found {element.label} at {element.bounding_box}")
"""

from typing import List

# Base classes and data models
from .analysis_base import (
    AnalysisInput,
    AnalysisResult,
    AnalysisType,
    BaseAnalyzer,
    BoundingBox,
    DetectedElement,
)

# Button detectors
from .button_color_detector import ButtonColorDetector
from .button_ensemble_detector import ButtonEnsembleDetector
from .button_fusion_detector import ButtonFusionDetector
from .button_hover_detector import ButtonHoverDetector
from .button_shadow_detector import ButtonShadowDetector
from .button_shape_detector import ButtonShapeDetector

# UI component detectors
from .dropdown_detector import DropdownDetector
from .icon_button_detector import IconButtonDetector
from .input_field_detector import InputFieldDetector
from .menu_bar_detector import MenuBarDetector
from .modal_dialog_detector import ModalDialogDetector
from .sidebar_detector import SidebarDetector
from .typography_detector import TypographyDetector

__all__: list[str] = [
    # Base classes
    "AnalysisInput",
    "AnalysisResult",
    "AnalysisType",
    "BaseAnalyzer",
    "BoundingBox",
    "DetectedElement",
    # Button detectors
    "ButtonColorDetector",
    "ButtonEnsembleDetector",
    "ButtonFusionDetector",
    "ButtonHoverDetector",
    "ButtonShadowDetector",
    "ButtonShapeDetector",
    # UI component detectors
    "DropdownDetector",
    "IconButtonDetector",
    "InputFieldDetector",
    "MenuBarDetector",
    "ModalDialogDetector",
    "SidebarDetector",
    "TypographyDetector",
]

__version__ = "0.1.0"
