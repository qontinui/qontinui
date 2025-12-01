"""Element Identifier module for qontinui library.

This module identifies and classifies StateImages, StateRegions, and StateLocations from
screenshots using computer vision techniques. It detects functional regions, classifies
element types, and understands spatial relationships within UI layouts.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np

from qontinui.discovery.models import StateImage


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
    """

    def __init__(self):
        """Initialize the element identifier with default parameters."""
        # Detection thresholds
        self.min_region_size = (20, 20)
        self.max_region_size = (2000, 2000)
        self.edge_threshold = 50
        self.line_threshold = 100
        self.grid_cell_min_size = 10
        self.grid_min_cells = 4

        # Classification thresholds
        self.button_aspect_ratio_range = (0.3, 4.0)
        self.icon_size_range = (16, 128)
        self.title_bar_height_range = (20, 60)

        # Spatial analysis
        self.alignment_tolerance = 5  # pixels
        self.adjacency_tolerance = 10  # pixels

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
        width = state_image.width
        height = state_image.height
        aspect_ratio = width / height if height > 0 else 0

        # Extract properties for classification
        properties = self._extract_element_properties(state_image, screenshot)

        # Window controls (small, square/round, in top corner)
        if self._is_window_control(state_image, properties):
            return ElementType.WINDOW_CONTROL

        # Icons (small, roughly square)
        if width <= self.icon_size_range[1] and height <= self.icon_size_range[1]:
            if 0.7 <= aspect_ratio <= 1.3:
                return ElementType.ICON

        # Buttons (medium size, reasonable aspect ratio)
        if self._is_button_like(state_image, properties, aspect_ratio):
            return ElementType.BUTTON

        # Text fields (elongated horizontally, specific characteristics)
        if aspect_ratio > 2.5 and self._has_text_field_characteristics(properties):
            return ElementType.TEXT_FIELD

        # Dividers (very thin in one dimension)
        if width < 5 or height < 5:
            return ElementType.DIVIDER

        # Text (based on OCR or text-like patterns)
        if self._appears_to_be_text(properties):
            return ElementType.TEXT

        # Checkboxes and radio buttons (small, square)
        if 10 <= width <= 30 and 10 <= height <= 30 and 0.8 <= aspect_ratio <= 1.2:
            if self._has_checkbox_characteristics(properties):
                return ElementType.CHECKBOX

        # Logo (larger, often in specific positions)
        if self._is_likely_logo(state_image, properties):
            return ElementType.LOGO

        return ElementType.UNKNOWN

    def detect_grid_regions(self, screenshot: np.ndarray) -> list[IdentifiedRegion]:
        """Detect grid patterns in the screenshot.

        Uses Hough line detection and pattern analysis to identify
        grid structures like inventory slots, skill bars, or tile layouts.

        Args:
            screenshot: Screenshot image

        Returns:
            List of detected grid regions
        """
        grid_regions = []
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.line_threshold,
            minLineLength=30,
            maxLineGap=10,
        )

        if lines is None:
            return grid_regions

        # Separate horizontal and vertical lines
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 10 or angle > 170:  # Horizontal
                h_lines.append((y1, x1, x2))  # (y, x_start, x_end)
            elif 80 < angle < 100:  # Vertical
                v_lines.append((x1, y1, y2))  # (x, y_start, y_end)

        # Cluster parallel lines
        h_clusters = self._cluster_parallel_lines(h_lines)
        v_clusters = self._cluster_parallel_lines(v_lines)

        # Find grid patterns where lines intersect regularly
        if len(h_clusters) >= 2 and len(v_clusters) >= 2:
            for h_group in h_clusters:
                for v_group in v_clusters:
                    if len(h_group) >= 2 and len(v_group) >= 2:
                        grid = self._extract_grid_region(h_group, v_group, screenshot)
                        if grid:
                            grid_regions.append(grid)

        return grid_regions

    def detect_panel_regions(self, screenshot: np.ndarray) -> list[IdentifiedRegion]:
        """Detect bordered panels or containers.

        Identifies rectangular regions with borders using contour detection
        and color analysis. Common in UI panels, dialogs, and containers.

        Args:
            screenshot: Screenshot image

        Returns:
            List of detected panel regions
        """
        panel_regions = []
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to find edges
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size
            if not self._is_valid_size((w, h)):
                continue

            # Check if it looks like a panel (rectangular, has area)
            area = cv2.contourArea(contour)
            bbox_area = w * h

            if bbox_area == 0:
                continue

            # Panel should be fairly rectangular
            rectangularity = area / bbox_area
            if rectangularity < 0.7:
                continue

            # Extract region for analysis
            region_img = screenshot[y : y + h, x : x + w]

            # Analyze if it has panel characteristics
            properties = self._analyze_panel_properties(region_img)

            if properties["is_panel"]:
                panel = IdentifiedRegion(
                    region_type=RegionType.PANEL,
                    bounds=(x, y, w, h),
                    confidence=properties["confidence"],
                    properties=properties,
                    sub_elements=[],
                )
                panel_regions.append(panel)

        return panel_regions

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
        title_bars = []
        height, width = screenshot.shape[:2]

        # Scan top portion of screen
        scan_height = min(height // 3, 200)

        # Look for horizontal bands with distinct colors
        for y in range(0, scan_height, 5):
            for bar_height in range(
                self.title_bar_height_range[0], self.title_bar_height_range[1], 5
            ):
                if y + bar_height >= height:
                    break

                # Extract potential title bar region
                region = screenshot[y : y + bar_height, 0:width]

                # Analyze characteristics
                if self._has_title_bar_characteristics(region):
                    # Look for actual width (might not span full screen)
                    actual_bounds = self._find_actual_title_bar_bounds(
                        screenshot, y, bar_height
                    )

                    if actual_bounds:
                        x, y_pos, w, h = actual_bounds
                        title_bar = IdentifiedRegion(
                            region_type=RegionType.TITLE_BAR,
                            bounds=(x, y_pos, w, h),
                            confidence=0.8,
                            properties={"scanned_y": y, "height": bar_height},
                            sub_elements=[],
                        )
                        title_bars.append(title_bar)

        # Remove overlapping/duplicate title bars
        title_bars = self._remove_overlapping_regions(title_bars)

        return title_bars

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
        relationships = []

        for i, elem1 in enumerate(elements):
            for _j, elem2 in enumerate(elements[i + 1 :], start=i + 1):
                # Check for various relationship types
                rel = self._analyze_element_pair(elem1, elem2)
                if rel:
                    relationships.append(rel)

        return relationships

    def _extract_element_properties(
        self, state_image: StateImage, screenshot: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Extract visual and contextual properties of an element.

        Args:
            state_image: StateImage to analyze
            screenshot: Optional full screenshot for context

        Returns:
            Dictionary of element properties
        """
        properties = {
            "width": state_image.width,
            "height": state_image.height,
            "aspect_ratio": (
                state_image.width / state_image.height if state_image.height > 0 else 0
            ),
            "area": state_image.width * state_image.height,
            "position": (state_image.x, state_image.y),
            "frequency": state_image.frequency,
            "mask_density": state_image.mask_density,
        }

        # Add pixel data properties if available
        if state_image.pixel_data is not None:
            properties.update(self._analyze_pixel_data(state_image.pixel_data))

        # Add position-based properties
        if screenshot is not None:
            screen_height, screen_width = screenshot.shape[:2]
            properties["relative_x"] = state_image.x / screen_width
            properties["relative_y"] = state_image.y / screen_height
            properties["in_top_region"] = state_image.y < screen_height * 0.15
            properties["in_corner"] = self._is_in_corner(
                state_image, screen_width, screen_height
            )

        return properties

    def _analyze_pixel_data(self, pixel_data: np.ndarray) -> dict[str, Any]:
        """Analyze pixel data to extract visual features.

        Args:
            pixel_data: Pixel data array

        Returns:
            Dictionary of visual features
        """
        features = {}

        # Color analysis
        if len(pixel_data.shape) == 3:
            mean_color = np.mean(pixel_data, axis=(0, 1))
            std_color = np.std(pixel_data, axis=(0, 1))
            features["mean_color"] = mean_color.tolist()
            features["std_color"] = std_color.tolist()
            features["color_variance"] = np.mean(std_color)

            # Convert to grayscale for edge analysis
            gray = cv2.cvtColor(pixel_data, cv2.COLOR_BGR2GRAY)
        else:
            gray = pixel_data
            features["color_variance"] = np.std(gray)

        # Edge density (indicates complexity)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features["edge_density"] = float(edge_density)

        # Texture analysis (using local standard deviation)
        kernel_size = 5
        if gray.shape[0] > kernel_size and gray.shape[1] > kernel_size:
            blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            texture = np.std(gray - blurred)
            features["texture"] = float(texture)
        else:
            features["texture"] = 0.0

        return features

    def _is_window_control(
        self, state_image: StateImage, properties: dict[str, Any]
    ) -> bool:
        """Check if element appears to be a window control button.

        Args:
            state_image: StateImage to check
            properties: Extracted properties

        Returns:
            True if likely a window control
        """
        # Small size
        if not (10 <= state_image.width <= 40 and 10 <= state_image.height <= 40):
            return False

        # In top corner or top-right area
        if not properties.get("in_top_region", False):
            return False

        # Roughly square or circular
        aspect_ratio = properties["aspect_ratio"]
        if not (0.7 <= aspect_ratio <= 1.3):
            return False

        # High frequency (appears consistently)
        if state_image.frequency < 0.7:
            return False

        return True

    def _is_button_like(
        self, state_image: StateImage, properties: dict[str, Any], aspect_ratio: float
    ) -> bool:
        """Check if element has button-like characteristics.

        Args:
            state_image: StateImage to check
            properties: Extracted properties
            aspect_ratio: Width/height ratio

        Returns:
            True if likely a button
        """
        # Reasonable size for a button
        if state_image.width < 30 or state_image.height < 15:
            return False

        # Button-like aspect ratio
        if not (
            self.button_aspect_ratio_range[0]
            <= aspect_ratio
            <= self.button_aspect_ratio_range[1]
        ):
            return False

        # Has some texture/content
        if properties.get("edge_density", 0) < 0.01:
            return False

        return True

    def _has_text_field_characteristics(self, properties: dict[str, Any]) -> bool:
        """Check if element has text field characteristics.

        Args:
            properties: Element properties

        Returns:
            True if likely a text field
        """
        # Text fields typically have low edge density (empty or simple text)
        edge_density = properties.get("edge_density", 1.0)
        if edge_density > 0.3:
            return False

        # Usually have a border or distinct background
        color_variance = properties.get("color_variance", 0)
        if color_variance > 50:  # Too much variation
            return False

        return True

    def _appears_to_be_text(self, properties: dict[str, Any]) -> bool:
        """Check if element appears to contain text.

        Args:
            properties: Element properties

        Returns:
            True if likely text
        """
        # Text has moderate edge density
        edge_density = properties.get("edge_density", 0)
        if not (0.05 <= edge_density <= 0.4):
            return False

        # Text usually has some texture
        texture = properties.get("texture", 0)
        if texture < 5:
            return False

        return True

    def _has_checkbox_characteristics(self, properties: dict[str, Any]) -> bool:
        """Check if element has checkbox characteristics.

        Args:
            properties: Element properties

        Returns:
            True if likely a checkbox
        """
        # Checkboxes have simple geometry
        edge_density = properties.get("edge_density", 0)

        # Should have clear edges (border) but not too complex
        return 0.1 <= edge_density <= 0.5  # type: ignore[no-any-return]

    def _is_likely_logo(
        self, state_image: StateImage, properties: dict[str, Any]
    ) -> bool:
        """Check if element is likely a logo.

        Args:
            state_image: StateImage to check
            properties: Element properties

        Returns:
            True if likely a logo
        """
        # Logos are often in top-left or have high visibility
        if not properties.get("in_top_region", False):
            return False

        # Reasonable logo size
        if state_image.width < 50 or state_image.height < 20:
            return False

        # High frequency (consistent across screenshots)
        if state_image.frequency < 0.8:
            return False

        # Often has higher visual complexity
        edge_density = properties.get("edge_density", 0)
        if edge_density < 0.1:
            return False

        return True

    def _is_in_corner(
        self, state_image: StateImage, screen_width: int, screen_height: int
    ) -> bool:
        """Check if element is in a screen corner.

        Args:
            state_image: StateImage to check
            screen_width: Screen width
            screen_height: Screen height

        Returns:
            True if in corner
        """
        corner_threshold = 0.1  # 10% from edges

        x_left = state_image.x < screen_width * corner_threshold
        x_right = state_image.x2 > screen_width * (1 - corner_threshold)
        y_top = state_image.y < screen_height * corner_threshold
        y_bottom = state_image.y2 > screen_height * (1 - corner_threshold)

        return (x_left or x_right) and (y_top or y_bottom)

    def _cluster_parallel_lines(
        self, lines: list[tuple], tolerance: int = 10
    ) -> list[list[tuple]]:
        """Cluster parallel lines that are close together.

        Args:
            lines: List of line coordinates
            tolerance: Distance tolerance for clustering

        Returns:
            List of line clusters
        """
        if not lines:
            return []

        # Sort lines by position
        sorted_lines = sorted(lines)
        clusters = [[sorted_lines[0]]]

        for line in sorted_lines[1:]:
            # Check if close to last cluster
            if abs(line[0] - clusters[-1][-1][0]) <= tolerance:
                clusters[-1].append(line)
            else:
                clusters.append([line])

        # Filter clusters with at least 2 lines
        return [cluster for cluster in clusters if len(cluster) >= 2]

    def _extract_grid_region(
        self,
        h_lines: list[tuple],
        v_lines: list[tuple],
        screenshot: np.ndarray,
    ) -> IdentifiedRegion | None:
        """Extract a grid region from horizontal and vertical line clusters.

        Args:
            h_lines: Horizontal lines
            v_lines: Vertical lines
            screenshot: Full screenshot

        Returns:
            IdentifiedRegion if valid grid found, None otherwise
        """
        # Calculate bounding box
        h_positions = [line[0] for line in h_lines]
        v_positions = [line[0] for line in v_lines]

        y_min, y_max = min(h_positions), max(h_positions)
        x_min, x_max = min(v_positions), max(v_positions)

        width = x_max - x_min
        height = y_max - y_min

        # Validate grid size
        if not self._is_valid_size((width, height)):
            return None

        # Calculate grid properties
        rows = len(h_lines) - 1
        cols = len(v_lines) - 1

        if rows < 1 or cols < 1:
            return None

        cell_width = width / cols if cols > 0 else 0
        cell_height = height / rows if rows > 0 else 0

        # Cells should be reasonable size
        if (
            cell_width < self.grid_cell_min_size
            or cell_height < self.grid_cell_min_size
        ):
            return None

        properties = {
            "rows": rows,
            "cols": cols,
            "cell_width": float(cell_width),
            "cell_height": float(cell_height),
            "regularity": self._calculate_grid_regularity(h_lines, v_lines),
        }

        return IdentifiedRegion(
            region_type=RegionType.GRID,
            bounds=(x_min, y_min, width, height),
            confidence=0.85,
            properties=properties,
            sub_elements=[],
        )

    def _calculate_grid_regularity(
        self, h_lines: list[tuple], v_lines: list[tuple]
    ) -> float:
        """Calculate how regular/uniform a grid pattern is.

        Args:
            h_lines: Horizontal lines
            v_lines: Vertical lines

        Returns:
            Regularity score (0.0 to 1.0)
        """
        # Check spacing consistency
        h_spacings = [
            h_lines[i + 1][0] - h_lines[i][0] for i in range(len(h_lines) - 1)
        ]
        v_spacings = [
            v_lines[i + 1][0] - v_lines[i][0] for i in range(len(v_lines) - 1)
        ]

        if not h_spacings or not v_spacings:
            return 0.0

        # Lower variance = more regular
        h_std = np.std(h_spacings) if len(h_spacings) > 1 else 0
        v_std = np.std(v_spacings) if len(v_spacings) > 1 else 0
        h_mean = np.mean(h_spacings)
        v_mean = np.mean(v_spacings)

        if h_mean == 0 or v_mean == 0:
            return 0.0

        # Coefficient of variation (lower is better)
        h_cv = h_std / h_mean
        v_cv = v_std / v_mean

        # Convert to regularity score (inverse of CV, clamped)
        avg_cv = (h_cv + v_cv) / 2
        regularity = max(0.0, 1.0 - avg_cv)

        return float(regularity)

    def _analyze_panel_properties(self, region_img: np.ndarray) -> dict[str, Any]:
        """Analyze if a region has panel characteristics.

        Args:
            region_img: Region image to analyze

        Returns:
            Dictionary with is_panel flag and confidence
        """
        properties = {"is_panel": False, "confidence": 0.0}

        if region_img.size == 0:
            return properties

        # Check for border (edges around perimeter)
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Check perimeter edge density
        h, w = edges.shape
        if h < 4 or w < 4:
            return properties

        border_thickness = 2
        top_border = edges[0:border_thickness, :]
        bottom_border = edges[h - border_thickness : h, :]
        left_border = edges[:, 0:border_thickness]
        right_border = edges[:, w - border_thickness : w]

        border_density = (
            np.sum(top_border > 0)
            + np.sum(bottom_border > 0)
            + np.sum(left_border > 0)
            + np.sum(right_border > 0)
        ) / (2 * border_thickness * (h + w))

        # Panels typically have visible borders
        if border_density > 0.3:
            properties["is_panel"] = True
            properties["confidence"] = min(0.9, border_density)  # type: ignore[assignment]
            properties["border_density"] = float(border_density)  # type: ignore[assignment]

        return properties

    def _has_title_bar_characteristics(self, region: np.ndarray) -> bool:
        """Check if region has title bar characteristics.

        Args:
            region: Region image to check

        Returns:
            True if likely a title bar
        """
        if region.size == 0:
            return False

        h, w = region.shape[:2]

        # Title bars are elongated horizontally
        if w < h * 3:
            return False

        # Check for uniform color (typical for title bars)
        std_color = np.std(region, axis=(0, 1))
        color_uniformity = np.mean(std_color)

        # Title bars often have low color variance
        if color_uniformity > 40:
            return False

        return True

    def _find_actual_title_bar_bounds(
        self, screenshot: np.ndarray, y: int, height: int
    ) -> tuple[int, int, int, int] | None:
        """Find actual bounds of a title bar (may not span full width).

        Args:
            screenshot: Full screenshot
            y: Y position to search
            height: Expected height

        Returns:
            Bounding box (x, y, w, h) or None
        """
        screen_height, screen_width = screenshot.shape[:2]

        # For now, return full width
        # TODO: Implement more sophisticated detection of actual title bar edges
        if y + height > screen_height:
            return None

        return (0, y, screen_width, height)

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

        keep: list[Any] = []
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

    def _is_valid_size(self, size: tuple[int, int]) -> bool:
        """Check if size is within valid range.

        Args:
            size: (width, height) tuple

        Returns:
            True if size is valid
        """
        w, h = size
        min_w, min_h = self.min_region_size
        max_w, max_h = self.max_region_size

        return min_w <= w <= max_w and min_h <= h <= max_h  # type: ignore[no-any-return]

    def _analyze_element_pair(
        self, elem1: IdentifiedElement, elem2: IdentifiedElement
    ) -> SpatialRelationship | None:
        """Analyze spatial relationship between two elements.

        Args:
            elem1: First element
            elem2: Second element

        Returns:
            SpatialRelationship if significant relationship found
        """
        x1, y1, w1, h1 = elem1.bounds
        x2, y2, w2, h2 = elem2.bounds

        # Calculate distance between centers
        center1 = (x1 + w1 / 2, y1 + h1 / 2)
        center2 = (x2 + w2 / 2, y2 + h2 / 2)
        distance = np.sqrt(
            (center2[0] - center1[0]) ** 2 + (center2[1] - center1[1]) ** 2
        )

        properties = {}
        relationship = "none"

        # Check alignment
        if abs(y1 - y2) <= self.alignment_tolerance:
            relationship = "horizontally_aligned"
            properties["alignment"] = "horizontal"
        elif abs(x1 - x2) <= self.alignment_tolerance:
            relationship = "vertically_aligned"
            properties["alignment"] = "vertical"

        # Check adjacency
        if abs((x1 + w1) - x2) <= self.adjacency_tolerance:
            relationship = "adjacent_left"
            properties["adjacency"] = "left"
        elif abs(x1 - (x2 + w2)) <= self.adjacency_tolerance:
            relationship = "adjacent_right"
            properties["adjacency"] = "right"
        elif abs((y1 + h1) - y2) <= self.adjacency_tolerance:
            relationship = "adjacent_above"
            properties["adjacency"] = "above"
        elif abs(y1 - (y2 + h2)) <= self.adjacency_tolerance:
            relationship = "adjacent_below"
            properties["adjacency"] = "below"

        # Check containment
        if x1 <= x2 and y1 <= y2 and x1 + w1 >= x2 + w2 and y1 + h1 >= y2 + h2:
            relationship = "contains"
            properties["containment"] = "contains"
        elif x2 <= x1 and y2 <= y1 and x2 + w2 >= x1 + w1 and y2 + h2 >= y1 + h1:
            relationship = "contained_by"
            properties["containment"] = "contained_by"

        # Only return if we found a meaningful relationship
        if relationship != "none":
            # Generate IDs from element bounds (since IdentifiedElement doesn't have ID)
            elem1_id = f"elem_{x1}_{y1}_{w1}_{h1}"
            elem2_id = f"elem_{x2}_{y2}_{w2}_{h2}"

            return SpatialRelationship(
                element1_id=elem1_id,
                element2_id=elem2_id,
                relationship=relationship,
                distance=float(distance),
                properties=properties,
            )

        return None
