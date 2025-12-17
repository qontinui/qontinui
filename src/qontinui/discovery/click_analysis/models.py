"""Data models for click analysis."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ElementType(Enum):
    """Types of GUI elements that can be detected at click locations."""

    BUTTON = "button"
    ICON = "icon"
    TEXT = "text"
    IMAGE = "image"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    INPUT_FIELD = "input_field"
    LINK = "link"
    MENU_ITEM = "menu_item"
    TAB = "tab"
    UNKNOWN = "unknown"


class DetectionStrategy(Enum):
    """Strategies for detecting element boundaries."""

    EDGE_BASED = "edge_based"  # Uses Canny edge detection
    CONTOUR_BASED = "contour_based"  # Uses contour detection
    COLOR_SEGMENTATION = "color_segmentation"  # Uses color uniformity
    FLOOD_FILL = "flood_fill"  # Uses flood fill from click point
    GRADIENT_BASED = "gradient_based"  # Uses gradient analysis
    TEMPLATE_MATCH = "template_match"  # Uses existing templates if available
    FIXED_SIZE = "fixed_size"  # Fallback to fixed-size box


@dataclass
class InferenceConfig:
    """Configuration for bounding box inference."""

    # Search area around click point
    search_radius: int = 100  # Max distance from click to search for boundaries

    # Size constraints for detected elements
    min_element_size: tuple[int, int] = (10, 10)
    max_element_size: tuple[int, int] = (500, 500)

    # Detection thresholds
    edge_threshold_low: int = 50
    edge_threshold_high: int = 150
    color_tolerance: int = 30
    contour_area_min: int = 100

    # Fallback settings
    fallback_box_size: int = 50  # Size when no element detected
    use_fallback: bool = True

    # Strategy preferences (in order of preference)
    preferred_strategies: list[DetectionStrategy] = field(
        default_factory=lambda: [
            DetectionStrategy.CONTOUR_BASED,
            DetectionStrategy.EDGE_BASED,
            DetectionStrategy.COLOR_SEGMENTATION,
            DetectionStrategy.FLOOD_FILL,
            DetectionStrategy.FIXED_SIZE,
        ]
    )

    # Advanced options
    enable_mask_generation: bool = True
    enable_element_classification: bool = True
    merge_nearby_boundaries: bool = True
    merge_gap: int = 5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "search_radius": self.search_radius,
            "min_element_size": self.min_element_size,
            "max_element_size": self.max_element_size,
            "edge_threshold_low": self.edge_threshold_low,
            "edge_threshold_high": self.edge_threshold_high,
            "color_tolerance": self.color_tolerance,
            "contour_area_min": self.contour_area_min,
            "fallback_box_size": self.fallback_box_size,
            "use_fallback": self.use_fallback,
            "preferred_strategies": [s.value for s in self.preferred_strategies],
            "enable_mask_generation": self.enable_mask_generation,
            "enable_element_classification": self.enable_element_classification,
            "merge_nearby_boundaries": self.merge_nearby_boundaries,
            "merge_gap": self.merge_gap,
        }


@dataclass
class InferredBoundingBox:
    """Result of bounding box inference from a click location."""

    # Bounding box coordinates (x, y, width, height)
    x: int
    y: int
    width: int
    height: int

    # Confidence in the detection (0.0 - 1.0)
    confidence: float

    # Detection metadata
    strategy_used: DetectionStrategy
    element_type: ElementType = ElementType.UNKNOWN

    # Optional mask for non-rectangular elements
    mask: np.ndarray[Any, Any] | None = None

    # Pixel data of the detected element
    pixel_data: np.ndarray[Any, Any] | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def x2(self) -> int:
        """Right edge x coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom edge y coordinate."""
        return self.y + self.height

    @property
    def center(self) -> tuple[int, int]:
        """Center point of the bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """Area of the bounding box."""
        return self.width * self.height

    def as_bbox_list(self) -> list[int]:
        """Return as [x, y, width, height] list for COCO format."""
        return [self.x, self.y, self.width, self.height]

    def contains_point(self, px: int, py: int) -> bool:
        """Check if a point is within this bounding box."""
        return self.x <= px < self.x2 and self.y <= py < self.y2

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "strategy_used": self.strategy_used.value,
            "element_type": self.element_type.value,
            "has_mask": self.mask is not None,
            "area": self.area,
            "metadata": self.metadata,
        }
        return result


@dataclass
class InferenceResult:
    """Complete result from click analysis."""

    # The click location that was analyzed
    click_location: tuple[int, int]

    # Primary inferred bounding box
    primary_bbox: InferredBoundingBox

    # Alternative candidates (other possible elements near click)
    alternative_candidates: list[InferredBoundingBox] = field(default_factory=list)

    # Image dimensions
    image_width: int = 0
    image_height: int = 0

    # Processing info
    strategies_attempted: list[DetectionStrategy] = field(default_factory=list)
    processing_time_ms: float = 0.0

    # Whether fallback was used
    used_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "click_location": self.click_location,
            "primary_bbox": self.primary_bbox.to_dict(),
            "alternative_candidates": [
                c.to_dict() for c in self.alternative_candidates
            ],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "strategies_attempted": [s.value for s in self.strategies_attempted],
            "processing_time_ms": self.processing_time_ms,
            "used_fallback": self.used_fallback,
        }
