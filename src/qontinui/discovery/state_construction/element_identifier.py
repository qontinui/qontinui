"""Element Identifier module for qontinui library.

This module identifies and classifies StateImages, StateRegions, and StateLocations from
screenshots using computer vision techniques. It detects functional regions, classifies
element types, and understands spatial relationships within UI layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from qontinui.discovery.models import StateImage

if TYPE_CHECKING:
    from qontinui.discovery.state_construction.element_classifier import (
        ElementClassifier,
    )
    from qontinui.discovery.state_construction.grid_detector import GridDetector
    from qontinui.discovery.state_construction.panel_detector import PanelDetector
    from qontinui.discovery.state_construction.spatial_analyzer import SpatialAnalyzer
    from qontinui.discovery.state_construction.title_bar_detector import (
        TitleBarDetector,
    )


class ElementType(Enum):
    """Types of UI elements that can be identified."""

    BUTTON = "button"
    ICON = "icon"
    TEXT = "text"
    TEXT_FIELD = "text_field"
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    DROPDOWN = "dropdown"
    SLIDER = "slider"
    IMAGE = "image"
    LOGO = "logo"
    DIVIDER = "divider"
    WINDOW_CONTROL = "window_control"
    MENU_ITEM = "menu_item"
    UNKNOWN = "unknown"


class RegionType(Enum):
    """Types of functional regions that can be identified."""

    GRID = "grid"
    PANEL = "panel"
    TITLE_BAR = "title_bar"
    NAVIGATION = "navigation"
    CONTENT = "content"
    TOOLBAR = "toolbar"
    SIDEBAR = "sidebar"
    FOOTER = "footer"
    MODAL = "modal"
    TOOLTIP = "tooltip"
    UNKNOWN = "unknown"


@dataclass
class IdentifiedRegion:
    """Represents an identified functional region.

    Attributes:
        region_type: Type of region (grid, panel, etc.)
        bounds: Bounding box (x, y, width, height)
        confidence: Confidence score (0.0 to 1.0)
        properties: Additional region properties
        sub_elements: List of elements within this region
    """

    region_type: RegionType
    bounds: tuple[int, int, int, int]
    confidence: float
    properties: dict[str, Any]
    sub_elements: list["IdentifiedElement"]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "region_type": self.region_type.value,
            "bounds": self.bounds,
            "confidence": self.confidence,
            "properties": self.properties,
            "sub_elements_count": len(self.sub_elements),
        }


@dataclass
class IdentifiedElement:
    """Represents an identified UI element with classification.

    Attributes:
        element_type: Type of element (button, icon, etc.)
        bounds: Bounding box (x, y, width, height)
        confidence: Classification confidence (0.0 to 1.0)
        properties: Additional element properties
        state_image: Optional associated StateImage
    """

    element_type: ElementType
    bounds: tuple[int, int, int, int]
    confidence: float
    properties: dict[str, Any]
    state_image: StateImage | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "element_type": self.element_type.value,
            "bounds": self.bounds,
            "confidence": self.confidence,
            "properties": self.properties,
        }
        if self.state_image:
            result["state_image_id"] = self.state_image.id
        return result


@dataclass
class SpatialRelationship:
    """Describes spatial relationship between elements.

    Attributes:
        element1_id: ID of first element
        element2_id: ID of second element
        relationship: Type of relationship (adjacent, contains, aligned, etc.)
        distance: Distance between elements in pixels
        properties: Additional relationship properties
    """

    element1_id: str
    element2_id: str
    relationship: str
    distance: float
    properties: dict[str, Any]


class ElementIdentifier:
    """Identifies and classifies UI elements and regions from screenshots.

    This class uses computer vision techniques to:
    - Identify functional regions (grids, panels, title bars)
    - Classify element types (buttons, icons, text fields)
    - Detect patterns (grids, aligned elements)
    - Analyze spatial relationships

    This is a facade class that delegates to specialized detector components.
    """

    _grid_detector: GridDetector
    _panel_detector: PanelDetector
    _title_bar_detector: TitleBarDetector
    _element_classifier: ElementClassifier
    _spatial_analyzer: SpatialAnalyzer

    def __init__(self) -> None:
        """Initialize the element identifier with default parameters."""
        # Import specialized detectors
        from qontinui.discovery.state_construction.element_classifier import (
            ElementClassifier,
        )
        from qontinui.discovery.state_construction.grid_detector import GridDetector
        from qontinui.discovery.state_construction.panel_detector import PanelDetector
        from qontinui.discovery.state_construction.spatial_analyzer import (
            SpatialAnalyzer,
        )
        from qontinui.discovery.state_construction.title_bar_detector import (
            TitleBarDetector,
        )

        # Detection thresholds (for backward compatibility)
        self.min_region_size = (20, 20)
        self.max_region_size = (2000, 2000)
        self.edge_threshold = 50
        self.line_threshold = 100
        self.grid_cell_min_size = 10
        self.grid_min_cells = 4

        # Classification thresholds (for backward compatibility)
        self.button_aspect_ratio_range = (0.3, 4.0)
        self.icon_size_range = (16, 128)
        self.title_bar_height_range = (20, 60)

        # Spatial analysis (for backward compatibility)
        self.alignment_tolerance = 5  # pixels
        self.adjacency_tolerance = 10  # pixels

        # Initialize specialized detectors
        self._grid_detector = GridDetector(
            line_threshold=self.line_threshold,
            grid_cell_min_size=self.grid_cell_min_size,
            grid_min_cells=self.grid_min_cells,
            min_region_size=self.min_region_size,
            max_region_size=self.max_region_size,
        )

        self._panel_detector = PanelDetector(
            min_region_size=self.min_region_size,
            max_region_size=self.max_region_size,
        )

        self._title_bar_detector = TitleBarDetector(
            title_bar_height_range=self.title_bar_height_range,
        )

        self._element_classifier = ElementClassifier(
            button_aspect_ratio_range=self.button_aspect_ratio_range,
            icon_size_range=self.icon_size_range,
            title_bar_height_range=self.title_bar_height_range,
        )

        self._spatial_analyzer = SpatialAnalyzer(
            alignment_tolerance=self.alignment_tolerance,
            adjacency_tolerance=self.adjacency_tolerance,
        )

    def identify_regions(
        self, screenshot: np.ndarray, state_images: list[StateImage] | None = None
    ) -> list[IdentifiedRegion]:
        """Identify functional regions in a screenshot.

        Detects various types of regions including grids, panels, title bars,
        and navigation areas using edge detection, contour analysis, and
        pattern recognition.

        Args:
            screenshot: Screenshot image as numpy array (BGR format)
            state_images: Optional list of StateImages to aid region detection

        Returns:
            List of identified regions with bounding boxes and types
        """
        regions = []

        # Detect different types of regions
        grid_regions = self.detect_grid_regions(screenshot)
        panel_regions = self.detect_panel_regions(screenshot)
        title_bar_regions = self.detect_title_bars(screenshot)

        regions.extend(grid_regions)
        regions.extend(panel_regions)
        regions.extend(title_bar_regions)

        # If state_images provided, use them to refine region detection
        if state_images:
            regions = self._refine_regions_with_elements(regions, state_images)

        # Remove overlapping regions (keep higher confidence)
        regions = self._remove_overlapping_regions(regions)

        return regions

    def classify_image_type(
        self, state_image: StateImage, screenshot: np.ndarray | None = None
    ) -> ElementType:
        """Classify the type of a StateImage element.

        Uses size, shape, visual features, and position to determine
        the likely element type (button, icon, text field, etc.).

        Args:
            state_image: StateImage to classify
            screenshot: Optional full screenshot for context

        Returns:
            Classified element type
        """
        return self._element_classifier.classify_image_type(state_image, screenshot)

    def detect_grid_regions(self, screenshot: np.ndarray) -> list[IdentifiedRegion]:
        """Detect grid patterns in the screenshot.

        Uses Hough line detection and pattern analysis to identify
        grid structures like inventory slots, skill bars, or tile layouts.

        Args:
            screenshot: Screenshot image

        Returns:
            List of detected grid regions
        """
        return self._grid_detector.detect_grid_regions(screenshot)

    def detect_panel_regions(self, screenshot: np.ndarray) -> list[IdentifiedRegion]:
        """Detect bordered panels or containers.

        Identifies rectangular regions with borders using contour detection
        and color analysis. Common in UI panels, dialogs, and containers.

        Args:
            screenshot: Screenshot image

        Returns:
            List of detected panel regions
        """
        return self._panel_detector.detect_panel_regions(screenshot)

    def detect_title_bars(self, screenshot: np.ndarray) -> list[IdentifiedRegion]:
        """Detect window title bars.

        Identifies title bars by looking for elongated horizontal regions
        at the top of panels, often with distinct background colors and
        containing text or window controls.

        Args:
            screenshot: Screenshot image

        Returns:
            List of detected title bar regions
        """
        return self._title_bar_detector.detect_title_bars(screenshot)

    def analyze_spatial_layout(
        self, elements: list[IdentifiedElement]
    ) -> list[SpatialRelationship]:
        """Analyze spatial relationships between elements.

        Identifies alignment, adjacency, containment, and other spatial
        relationships that help understand UI structure.

        Args:
            elements: List of identified elements

        Returns:
            List of spatial relationships between elements
        """
        return self._spatial_analyzer.analyze_spatial_layout(elements)

    def _refine_regions_with_elements(
        self, regions: list[IdentifiedRegion], state_images: list[StateImage]
    ) -> list[IdentifiedRegion]:
        """Refine region detection using known elements.

        Args:
            regions: Initially detected regions
            state_images: Known StateImages

        Returns:
            Refined list of regions
        """
        # Assign elements to regions based on spatial containment
        for region in regions:
            rx, ry, rw, rh = region.bounds
            for state_image in state_images:
                # Check if element is within region
                if (
                    rx <= state_image.x
                    and ry <= state_image.y
                    and state_image.x2 <= rx + rw
                    and state_image.y2 <= ry + rh
                ):
                    # Classify the element
                    element_type = self.classify_image_type(state_image)
                    identified_elem = IdentifiedElement(
                        element_type=element_type,
                        bounds=(
                            state_image.x,
                            state_image.y,
                            state_image.width,
                            state_image.height,
                        ),
                        confidence=0.7,
                        properties={},
                        state_image=state_image,
                    )
                    region.sub_elements.append(identified_elem)

        return regions

    def _remove_overlapping_regions(
        self, regions: list[IdentifiedRegion]
    ) -> list[IdentifiedRegion]:
        """Remove overlapping regions, keeping higher confidence ones.

        Args:
            regions: List of regions

        Returns:
            Filtered list without significant overlaps
        """
        if not regions:
            return []

        # Sort by confidence (descending)
        sorted_regions = sorted(regions, key=lambda r: r.confidence, reverse=True)

        keep: list[IdentifiedRegion] = []
        for region in sorted_regions:
            # Check overlap with kept regions
            overlaps = False
            for kept_region in keep:
                if self._calculate_region_overlap(region, kept_region) > 0.5:
                    overlaps = True
                    break

            if not overlaps:
                keep.append(region)

        return keep

    def _calculate_region_overlap(
        self, region1: IdentifiedRegion, region2: IdentifiedRegion
    ) -> float:
        """Calculate overlap between two regions.

        Args:
            region1: First region
            region2: Second region

        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        x1, y1, w1, h1 = region1.bounds
        x2, y2, w2, h2 = region2.bounds

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union if union > 0 else 0.0
