"""Approved Template data model.

Represents a user-approved template candidate from the web UI,
ready for state machine generation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from .models import InferredBoundingBox


@dataclass
class ApprovedTemplate:
    """Template approved by user in web UI for state machine generation.

    This represents a click-detected boundary that has been reviewed
    and approved by the user, potentially with manual adjustments.

    Attributes:
        id: Unique identifier for this template
        session_id: ID of the capture session this template came from
        click_x: X coordinate of the original click
        click_y: Y coordinate of the original click
        click_timestamp: Unix timestamp of when the click occurred
        frame_number: Frame number in the video where click occurred
        boundary: The approved bounding box (may be user-adjusted)
        mask: Optional mask for non-rectangular elements
        name: User-assigned name for this template
        state_hint: User-assigned state grouping hint
        element_type: Detected or user-specified element type
        pixel_data: Extracted pixel data for the template
        confidence: Detection confidence score
        approved_at: When the user approved this template
        metadata: Additional metadata
    """

    # Identity
    id: str
    session_id: str

    # Click context
    click_x: int
    click_y: int
    click_timestamp: float
    frame_number: int

    # Approved boundary (may be adjusted from auto-detected)
    boundary: InferredBoundingBox

    # Optional mask (user may have refined it)
    mask: np.ndarray | None = None

    # User-assigned properties
    name: str | None = None
    state_hint: str | None = None
    element_type: str = "unknown"

    # Template data
    pixel_data: np.ndarray | None = None
    confidence: float = 0.0

    # Timestamps
    approved_at: datetime = field(default_factory=datetime.now)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        """Get template width from boundary."""
        return self.boundary.width

    @property
    def height(self) -> int:
        """Get template height from boundary."""
        return self.boundary.height

    @property
    def center(self) -> tuple[int, int]:
        """Get center point of the boundary."""
        return self.boundary.center

    @property
    def bounding_box(self) -> tuple[int, int, int, int]:
        """Get bounding box as (x, y, width, height) tuple."""
        return (
            self.boundary.x,
            self.boundary.y,
            self.boundary.width,
            self.boundary.height,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "click_x": self.click_x,
            "click_y": self.click_y,
            "click_timestamp": self.click_timestamp,
            "frame_number": self.frame_number,
            "boundary": self.boundary.to_dict(),
            "mask": self.mask.tolist() if self.mask is not None else None,
            "name": self.name,
            "state_hint": self.state_hint,
            "element_type": self.element_type,
            "confidence": self.confidence,
            "approved_at": self.approved_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApprovedTemplate":
        """Create from dictionary."""
        boundary_data = data["boundary"]
        boundary = InferredBoundingBox(
            x=boundary_data["x"],
            y=boundary_data["y"],
            width=boundary_data["width"],
            height=boundary_data["height"],
            confidence=boundary_data.get("confidence", 0.0),
            strategy_used=boundary_data.get("strategy_used", "unknown"),
            element_type=boundary_data.get("element_type", "unknown"),
            mask=np.array(boundary_data["mask"]) if boundary_data.get("mask") else None,
            pixel_data=(
                np.array(boundary_data["pixel_data"]) if boundary_data.get("pixel_data") else None
            ),
            metadata=boundary_data.get("metadata", {}),
        )

        mask = np.array(data["mask"]) if data.get("mask") else None
        pixel_data = np.array(data["pixel_data"]) if data.get("pixel_data") else None

        approved_at = data.get("approved_at")
        if isinstance(approved_at, str):
            approved_at = datetime.fromisoformat(approved_at)
        elif approved_at is None:
            approved_at = datetime.now()

        return cls(
            id=data["id"],
            session_id=data["session_id"],
            click_x=data["click_x"],
            click_y=data["click_y"],
            click_timestamp=data["click_timestamp"],
            frame_number=data["frame_number"],
            boundary=boundary,
            mask=mask,
            name=data.get("name"),
            state_hint=data.get("state_hint"),
            element_type=data.get("element_type", "unknown"),
            pixel_data=pixel_data,
            confidence=data.get("confidence", 0.0),
            approved_at=approved_at,
            metadata=data.get("metadata", {}),
        )
