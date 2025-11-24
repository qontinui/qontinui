"""
Base classes and interfaces for region analysis modules

Region analysis focuses on detecting larger functional areas of the UI
(e.g., inventory grids, minimaps, toolbars) as opposed to individual elements.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID


class RegionType(str, Enum):
    """Types of regions that can be detected"""

    INVENTORY_GRID = "inventory_grid"  # Grid-based inventory systems
    MINIMAP = "minimap"  # Mini-maps in games
    STATUS_BAR = "status_bar"  # Status/health bars
    TOOLBAR = "toolbar"  # Tool/action bars
    DIALOG = "dialog"  # Dialog boxes/windows
    MENU = "menu"  # Menu areas
    CHAT_PANEL = "chat_panel"  # Chat/message panels
    SKILL_TREE = "skill_tree"  # Skill/ability trees
    QUEST_LOG = "quest_log"  # Quest/mission panels
    EQUIPMENT_PANEL = "equipment_panel"  # Character equipment
    TEXT_AREA = "text_area"  # Text regions (labels, captions, etc.)
    WINDOW = "window"  # Application windows with title bars
    TITLE_BAR = "title_bar"  # Window title bars
    CLOSE_BUTTON = "close_button"  # Window close buttons (X icon)
    CUSTOM = "custom"  # Extensibility for game-specific regions


class RegionAnalysisType(str, Enum):
    """Types of region analysis methods"""

    TEMPLATE_MATCH = "template_match"  # Template matching for known regions
    EDGE_DETECTION = "edge_detection"  # Edge-based region detection
    COLOR_CLUSTERING = "color_clustering"  # Color-based region segmentation
    PATTERN_ANALYSIS = "pattern_analysis"  # Pattern-based detection (grids, etc.)
    ML_CLASSIFICATION = "ml_classification"  # ML-based region classification
    CUSTOM = "custom"  # Extensibility for future methods


@dataclass
class BoundingBox:
    """Represents a bounding box for a region"""

    x: int
    y: int
    width: int
    height: int

    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union with another box"""
        # Calculate intersection
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = self.width * self.height + other.width * other.height - intersection

        return intersection / union if union > 0 else 0.0

    def overlaps(self, other: "BoundingBox", threshold: float = 0.5) -> bool:
        """Check if this box significantly overlaps with another"""
        return self.iou(other) >= threshold

    def contains(self, other: "BoundingBox") -> bool:
        """Check if this box completely contains another box"""
        return (
            self.x <= other.x
            and self.y <= other.y
            and self.x + self.width >= other.x + other.width
            and self.y + self.height >= other.y + other.height
        )

    def is_contained_by(self, other: "BoundingBox") -> bool:
        """Check if this box is completely contained by another box"""
        return other.contains(self)

    def area(self) -> int:
        """Calculate the area of this bounding box"""
        return self.width * self.height

    def center(self) -> tuple[int, int]:
        """Get the center point of this bounding box"""
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class DetectedRegion:
    """Represents a detected region in a screenshot"""

    bounding_box: BoundingBox
    confidence: float  # 0.0 to 1.0
    region_type: RegionType
    label: Optional[str] = None  # Optional descriptive label
    screenshot_index: int = 0  # Which screenshot in the set
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "bounding_box": {
                "x": self.bounding_box.x,
                "y": self.bounding_box.y,
                "width": self.bounding_box.width,
                "height": self.bounding_box.height,
            },
            "confidence": self.confidence,
            "region_type": self.region_type.value,
            "label": self.label,
            "screenshot_index": self.screenshot_index,
            "metadata": self.metadata,
        }


@dataclass
class RegionAnalysisResult:
    """Result from a region analysis method"""

    analyzer_type: RegionAnalysisType
    analyzer_name: str
    regions: List[DetectedRegion]
    confidence: float  # Overall confidence in this analysis
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "analyzer_type": self.analyzer_type.value,
            "analyzer_name": self.analyzer_name,
            "regions": [region.to_dict() for region in self.regions],
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class RegionAnalysisInput:
    """Input data for region analysis"""

    annotation_set_id: UUID
    screenshots: List[Dict[str, Any]]  # List of screenshot metadata
    screenshot_data: List[bytes]  # Actual image data
    parameters: Dict[str, Any] = field(default_factory=dict)


class BaseRegionAnalyzer(ABC):
    """
    Base class for all region analysis modules

    Each analyzer implements a specific method for detecting UI regions.
    Analyzers should be stateless and thread-safe.

    Key differences from element detection:
    - Regions are larger functional areas (inventory panels, minimaps)
    - Elements are individual UI components (buttons, inputs)
    - Regions may contain multiple elements
    - Region detection often uses different techniques (clustering, segmentation)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the region analyzer

        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}

    @property
    @abstractmethod
    def analysis_type(self) -> RegionAnalysisType:
        """Return the type of region analysis this module performs"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a unique name for this region analyzer"""
        pass

    @property
    def version(self) -> str:
        """Return the version of this analyzer"""
        return "1.0.0"

    @abstractmethod
    async def analyze(self, input_data: RegionAnalysisInput) -> RegionAnalysisResult:
        """
        Perform region analysis on the input data

        Args:
            input_data: Input screenshots and parameters

        Returns:
            RegionAnalysisResult containing detected regions
        """
        pass

    def validate_input(self, input_data: RegionAnalysisInput) -> bool:
        """
        Validate that input data is suitable for this analyzer

        Returns:
            True if input is valid, False otherwise
        """
        return True

    @property
    def required_screenshots(self) -> int:
        """
        Minimum number of screenshots required

        Returns:
            Minimum screenshots needed (0 = no minimum)
        """
        return 0

    @property
    def supports_multi_screenshot(self) -> bool:
        """Whether this analyzer can process multiple screenshots"""
        return False

    @property
    def supported_region_types(self) -> List[RegionType]:
        """
        List of region types this analyzer can detect

        Returns:
            List of RegionType enums
        """
        return [RegionType.CUSTOM]

    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get default parameters for this analyzer

        Returns:
            Dictionary of default parameter values
        """
        return {}
