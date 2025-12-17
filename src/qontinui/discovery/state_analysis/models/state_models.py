"""Data models for state detection and image extraction.

This module defines the data structures used throughout the state detection
and image extraction pipeline.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class InputEvent:
    """Represents a user input event (click, keypress, etc.)."""

    timestamp: float
    event_type: str  # 'click', 'key', 'move', etc.
    x: int | None = None
    y: int | None = None
    button: str | None = None
    key: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class Frame:
    """Represents a single frame/screenshot with metadata."""

    image: np.ndarray  # OpenCV image (BGR format)
    timestamp: float
    frame_index: int
    file_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateImage:
    """Represents an identifying image extracted from a state.

    StateImages are persistent visual elements that help identify and
    distinguish states. They are typically UI elements like buttons,
    icons, or labels that appear consistently within a state.
    """

    name: str
    image: np.ndarray  # The extracted image region (BGR format)
    bbox: tuple[int, int, int, int]  # (x, y, width, height)
    position_type: str  # "fixed" or "dynamic"
    position: tuple[int, int]  # (x, y) coordinates of top-left corner
    similarity_threshold: float = 0.85  # Match threshold for this image
    context_image: np.ndarray | None = None  # Larger context around the image
    source_frame_index: int | None = None  # Frame this was extracted from
    extraction_method: str = "manual"  # How this image was extracted
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary (excluding image data)."""
        return {
            "name": self.name,
            "bbox": self.bbox,
            "position_type": self.position_type,
            "position": self.position,
            "similarity_threshold": self.similarity_threshold,
            "source_frame_index": self.source_frame_index,
            "extraction_method": self.extraction_method,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateImage":
        """Create StateImage from dictionary.

        Args:
            data: Dictionary containing state image data (without image arrays)

        Returns:
            New StateImage instance with placeholder image arrays
        """
        # Create placeholder arrays since images can't be serialized
        placeholder_h, placeholder_w = data["bbox"][3], data["bbox"][2]
        placeholder_image = np.zeros((placeholder_h, placeholder_w, 3), dtype=np.uint8)

        return cls(
            name=data["name"],
            image=placeholder_image,
            bbox=tuple(data["bbox"]),
            position_type=data["position_type"],
            position=tuple(data["position"]),
            similarity_threshold=data.get("similarity_threshold", 0.85),
            context_image=None,
            source_frame_index=data.get("source_frame_index"),
            extraction_method=data.get("extraction_method", "manual"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DetectedState:
    """Represents a detected application state.

    A state is a distinct visual configuration of the application,
    characterized by specific UI elements, regions, and interaction points.
    """

    id: str
    name: str
    description: str
    state_images: list[StateImage]
    start_frame_index: int
    end_frame_index: int
    frame_indices: list[int] = field(default_factory=list)
    boundary: tuple[int, int, int, int] | None = None  # (x, y, width, height)
    click_locations: list[tuple[int, int]] = field(default_factory=list)
    transitions_to: list[str] = field(default_factory=list)  # State names
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "state_images": [img.to_dict() for img in self.state_images],
            "start_frame_index": self.start_frame_index,
            "end_frame_index": self.end_frame_index,
            "frame_indices": self.frame_indices,
            "boundary": self.boundary,
            "click_locations": self.click_locations,
            "transitions_to": self.transitions_to,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectedState":
        """Create DetectedState from dictionary.

        Args:
            data: Dictionary containing detected state data

        Returns:
            New DetectedState instance
        """
        # Convert boundary list to tuple if present
        boundary = data.get("boundary")
        if boundary is not None and not isinstance(boundary, tuple):
            boundary = tuple(boundary)

        # Convert click_locations to list of tuples
        click_locations = [
            tuple(loc) if not isinstance(loc, tuple) else loc
            for loc in data.get("click_locations", [])
        ]

        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            state_images=[StateImage.from_dict(img) for img in data["state_images"]],
            start_frame_index=data["start_frame_index"],
            end_frame_index=data["end_frame_index"],
            frame_indices=data.get("frame_indices", []),
            boundary=boundary,
            click_locations=click_locations,
            transitions_to=data.get("transitions_to", []),
            metadata=data.get("metadata", {}),
        )
