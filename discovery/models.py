"""
Data models for State Discovery
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AnalysisConfig:
    """Configuration for state discovery analysis"""

    min_region_size: tuple[int, int] = (20, 20)
    max_region_size: tuple[int, int] = (500, 500)
    color_tolerance: int = 5
    stability_threshold: float = 0.98
    variance_threshold: float = 10
    min_screenshots_present: int = 2
    processing_mode: str = "full"
    enable_rectangle_decomposition: bool = True
    enable_cooccurrence_analysis: bool = True

    # Pixel analysis thresholds
    dark_pixel_threshold: int = 60  # Pixels darker than this are considered dark
    light_pixel_threshold: int = 200  # Pixels lighter than this are considered light


@dataclass
class StateImage:
    """Represents a detected stable region that appears across screenshots"""

    id: str = field(default_factory=lambda: f"si_{uuid.uuid4().hex[:12]}")
    name: str = ""
    x: int = 0
    y: int = 0
    x2: int = 0
    y2: int = 0
    width: int = 0
    height: int = 0
    pixel_hash: str = ""
    stability_score: float = 1.0  # 1.0 = perfect pixel match when present
    screenshots: list[str] = field(default_factory=list)  # Screenshots where this appears
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: list[str] = field(default_factory=list)

    # Pixel percentage data
    dark_pixel_percentage: float = 0.0  # Percentage of dark pixels
    light_pixel_percentage: float = 0.0  # Percentage of light pixels
    avg_brightness: float = 0.0  # Average brightness (0-255)

    def to_dict(self) -> dict[str, Any]:
        import numpy as np

        def convert(val):
            """Convert numpy types to Python native types."""
            if isinstance(val, np.integer | np.int32 | np.int64):
                return int(val)
            elif isinstance(val, np.floating | np.float32 | np.float64):
                return float(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            return val

        return {
            "id": self.id,
            "name": self.name,
            "x": convert(self.x),
            "y": convert(self.y),
            "x2": convert(self.x2),
            "y2": convert(self.y2),
            "width": convert(self.width),
            "height": convert(self.height),
            "pixelHash": self.pixel_hash,
            "stabilityScore": convert(self.stability_score),
            "screenshots": self.screenshots,  # List of screenshots where found
            "createdAt": self.created_at,
            "tags": self.tags,
            "darkPixelPercentage": convert(self.dark_pixel_percentage),
            "lightPixelPercentage": convert(self.light_pixel_percentage),
            "avgBrightness": convert(self.avg_brightness),
        }


@dataclass
class DiscoveredState:
    """Represents a discovered application state"""

    id: str = field(default_factory=lambda: f"state_{uuid.uuid4().hex[:12]}")
    name: str = ""
    state_image_ids: list[str] = field(default_factory=list)
    screenshot_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        import numpy as np

        def convert(val):
            """Convert numpy types to Python native types."""
            if isinstance(val, np.integer | np.int32 | np.int64):
                return int(val)
            elif isinstance(val, np.floating | np.float32 | np.float64):
                return float(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            return val

        return {
            "id": self.id,
            "name": self.name,
            "stateImageIds": self.state_image_ids,
            "screenshotIds": self.screenshot_ids,
            "confidence": convert(self.confidence),
            "metadata": self.metadata,
        }


@dataclass
class AnalysisResult:
    """Results from state discovery analysis"""

    states: list[DiscoveredState] = field(default_factory=list)
    state_images: list[StateImage] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "states": [s.to_dict() for s in self.states],
            "state_images": [si.to_dict() for si in self.state_images],
            "transitions": self.transitions,
            "statistics": self.statistics,
        }


@dataclass
class DeleteOptions:
    """Options for deletion operations"""

    cascade: bool = False
    force: bool = False
    skip_critical: bool = False


@dataclass
class DeletionImpact:
    """Impact analysis for deletion"""

    affected_states: list[str] = field(default_factory=list)
    affected_transitions: list[str] = field(default_factory=list)
    is_critical: bool = False
    warning_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "affectedStates": self.affected_states,
            "affectedTransitions": self.affected_transitions,
            "isCritical": self.is_critical,
            "warningMessage": self.warning_message,
        }


@dataclass
class DeleteResult:
    """Result of deletion operation"""

    success: bool = False
    deleted_items: list[str] = field(default_factory=list)
    cascade_deletions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "deletedItems": self.deleted_items,
            "cascadeDeletions": self.cascade_deletions,
            "errors": self.errors,
        }
