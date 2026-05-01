"""Data models for State Discovery system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class StateImage:
    """Represents a discovered UI element that appears consistently across screenshots."""

    id: str
    name: str
    x: int
    y: int
    x2: int  # Bottom-right x coordinate
    y2: int  # Bottom-right y coordinate
    pixel_hash: str  # Hash of masked pixel data for comparison
    frequency: float  # Percentage of screenshots containing this element
    screenshot_ids: list[str] = field(default_factory=list)
    pixel_data: np.ndarray[Any, Any] | None = None
    mask: np.ndarray[Any, Any] | None = (
        None  # Mask array (0.0-1.0), shape (height, width)
    )
    created_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)
    dark_pixel_percentage: float | None = None  # Calculated only for active mask pixels
    light_pixel_percentage: float | None = (
        None  # Calculated only for active mask pixels
    )
    mask_density: float = (
        1.0  # Percentage of active pixels in mask (1.0 = full rectangle)
    )

    @property
    def width(self) -> int:
        """Calculate width from coordinates."""
        return self.x2 - self.x

    @property
    def height(self) -> int:
        """Calculate height from coordinates."""
        return self.y2 - self.y

    @property
    def center(self) -> tuple[Any, ...]:
        """Calculate center point."""
        return ((self.x + self.x2) // 2, (self.y + self.y2) // 2)

    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is within this StateImage's bounds."""
        return self.x <= x <= self.x2 and self.y <= y <= self.y2

    def overlaps(self, other: "StateImage") -> bool:
        """Check if this StateImage overlaps with another."""
        return not (
            self.x2 < other.x
            or other.x2 < self.x
            or self.y2 < other.y
            or other.y2 < self.y
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "pixel_hash": self.pixel_hash,
            "frequency": self.frequency,
            "screenshots": self.screenshot_ids,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
        }

        # Include pixel percentages if available
        if self.dark_pixel_percentage is not None:
            result["darkPixelPercentage"] = self.dark_pixel_percentage
        if self.light_pixel_percentage is not None:
            result["lightPixelPercentage"] = self.light_pixel_percentage

        # Include mask metadata
        result["maskDensity"] = self.mask_density
        result["hasMask"] = self.mask is not None

        return result


@dataclass
class DiscoveredState:
    """Represents a discovered application state composed of StateImages."""

    id: str
    name: str
    state_image_ids: list[str]
    screenshot_ids: list[str]
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "stateImageIds": self.state_image_ids,  # Changed to match frontend
            "screenshotIds": self.screenshot_ids,  # Changed to match frontend
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class StateTransition:
    """Represents a discovered transition between states."""

    from_state: str
    to_state: str
    trigger_image: str | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "trigger_image": self.trigger_image,
            "confidence": self.confidence,
        }


@dataclass
class AnalysisResult:
    """Results from state discovery analysis."""

    states: list[DiscoveredState]
    state_images: list[StateImage]
    transitions: list[StateTransition]
    stability_map: np.ndarray[Any, Any] | None = None
    statistics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "states": [s.to_dict() for s in self.states],
            "state_images": [si.to_dict() for si in self.state_images],
            "transitions": [t.to_dict() for t in self.transitions],
            "states_count": len(self.states),
            "state_images_count": len(self.state_images),
            "statistics": self.statistics,
        }


@dataclass
class AnalysisConfig:
    """Configuration for state discovery analysis."""

    min_region_size: tuple[Any, ...] = (20, 20)
    max_region_size: tuple[Any, ...] = (500, 500)
    color_tolerance: int = 5
    stability_threshold: float = 0.98
    variance_threshold: float = 10.0
    min_screenshots_present: int = 2
    enable_rectangle_decomposition: bool = True
    enable_cooccurrence_analysis: bool = True
    processing_mode: str = "full"  # 'full', 'quick', 'custom'
    similarity_threshold: float = (
        0.95  # Similarity threshold for matching regions (0.0-1.0)
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_region_size": self.min_region_size,
            "max_region_size": self.max_region_size,
            "color_tolerance": self.color_tolerance,
            "stability_threshold": self.stability_threshold,
            "variance_threshold": self.variance_threshold,
            "min_screenshots_present": self.min_screenshots_present,
            "enable_rectangle_decomposition": self.enable_rectangle_decomposition,
            "enable_cooccurrence_analysis": self.enable_cooccurrence_analysis,
            "processing_mode": self.processing_mode,
            "similarity_threshold": self.similarity_threshold,
        }


@dataclass
class DeletionImpact:
    """Impact analysis for StateImage deletion."""

    state_image: StateImage
    states_affected: int
    affected_state_ids: list[str]
    will_create_orphans: bool
    orphaned_state_ids: list[str]
    is_critical: bool
    is_frequently_used: bool
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "state_image": self.state_image.to_dict(),
            "states_affected": self.states_affected,
            "affected_states": self.affected_state_ids,
            "will_create_orphans": self.will_create_orphans,
            "orphaned_states": self.orphaned_state_ids,
            "is_critical": self.is_critical,
            "is_frequently_used": self.is_frequently_used,
            "recommendations": self.recommendations,
        }


@dataclass
class DeleteOptions:
    """Options for StateImage deletion."""

    cascade: bool = True
    force: bool = False
    skip_confirmation: bool = False
    handle_orphans: str = "keep"  # 'delete', 'keep', 'merge'

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cascade": self.cascade,
            "force": self.force,
            "skip_confirmation": self.skip_confirmation,
            "handle_orphans": self.handle_orphans,
        }


@dataclass
class DeleteResult:
    """Result of StateImage deletion."""

    deleted: list[str]
    skipped: list[dict[str, str]]
    affected_states: list[str]
    orphaned_states: list[str]
    warnings: list[str]
    undo_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "deleted": self.deleted,
            "skipped": self.skipped,
            "affected_states": self.affected_states,
            "orphaned_states": self.orphaned_states,
            "warnings": self.warnings,
            "undo_id": self.undo_id,
        }
