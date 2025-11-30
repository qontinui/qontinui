"""
Base classes and interfaces for analysis modules
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID


class AnalysisType(str, Enum):
    """Types of analysis methods"""

    STABLE_REGION = "stable_region"  # Type 1: Elements stable across screenshots
    PATTERN_MATCH = "pattern_match"  # Type 2: Same pattern in different positions
    SINGLE_SHOT = "single_shot"  # Type 3: ML/CV analysis of single screenshots
    CUSTOM = "custom"  # Extensibility for future methods


@dataclass
class BoundingBox:
    """Represents a detected element bounding box"""

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


@dataclass
class DetectedElement:
    """Represents a detected GUI element"""

    bounding_box: BoundingBox
    confidence: float  # 0.0 to 1.0
    label: str | None = None
    element_type: str | None = None  # button, input, image, etc.
    screenshot_index: int = 0  # Which screenshot in the set
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "bounding_box": {
                "x": self.bounding_box.x,
                "y": self.bounding_box.y,
                "width": self.bounding_box.width,
                "height": self.bounding_box.height,
            },
            "confidence": self.confidence,
            "label": self.label,
            "element_type": self.element_type,
            "screenshot_index": self.screenshot_index,
            "metadata": self.metadata,
        }


@dataclass
class AnalysisResult:
    """Result from an analysis method"""

    analyzer_type: AnalysisType
    analyzer_name: str
    elements: list[DetectedElement]
    confidence: float  # Overall confidence in this analysis
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "analyzer_type": self.analyzer_type.value,
            "analyzer_name": self.analyzer_name,
            "elements": [elem.to_dict() for elem in self.elements],
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class AnalysisInput:
    """Input data for analysis"""

    annotation_set_id: UUID
    screenshots: list[dict[str, Any]]  # List of screenshot metadata
    screenshot_data: list[bytes]  # Actual image data
    parameters: dict[str, Any] = field(default_factory=dict)


class BaseAnalyzer(ABC):
    """
    Base class for all analysis modules

    Each analyzer implements a specific method for detecting GUI elements.
    Analyzers should be stateless and thread-safe.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the analyzer

        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}

    @property
    @abstractmethod
    def analysis_type(self) -> AnalysisType:
        """Return the type of analysis this module performs"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a unique name for this analyzer"""
        pass

    @property
    def version(self) -> str:
        """Return the version of this analyzer"""
        return "1.0.0"

    @abstractmethod
    async def analyze(self, input_data: AnalysisInput) -> AnalysisResult:
        """
        Perform analysis on the input data

        Args:
            input_data: Input screenshots and parameters

        Returns:
            AnalysisResult containing detected elements
        """
        pass

    def validate_input(self, input_data: AnalysisInput) -> bool:
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

    def get_default_parameters(self) -> dict[str, Any]:
        """
        Get default parameters for this analyzer

        Returns:
            Dictionary of default parameter values
        """
        return {}
