"""Template candidate data model for click-to-template system.

A ClickTemplateCandidate represents a potential template extracted from a
captured click event during a recording session. It contains all information
needed for review, adjustment, and eventual import into a state machine.
"""

import base64
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from .models import DetectionStrategy, InferredBoundingBox


def _encode_array(arr: np.ndarray | None) -> str | None:
    """Encode numpy array to base64 string."""
    if arr is None:
        return None
    return base64.b64encode(arr.tobytes()).decode("utf-8")


def _decode_array(data: str | None, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray | None:
    """Decode base64 string to numpy array."""
    if data is None:
        return None
    return np.frombuffer(base64.b64decode(data), dtype=dtype).reshape(shape)


@dataclass
class ClickTemplateCandidate:
    """A template candidate extracted from a click event during capture.

    This represents a single clicked element that has been detected and
    extracted from a video frame. It contains both the detection results
    and the extracted pixel data, ready for review and potential import
    into a state machine configuration.

    Attributes:
        id: Unique identifier for this candidate.
        session_id: ID of the capture session this came from.
        click_x: X coordinate of the original click.
        click_y: Y coordinate of the original click.
        click_button: Mouse button used ('left', 'right', 'middle').
        timestamp: Time of the click in seconds from session start.
        frame_number: Video frame number where click occurred.
        primary_boundary: Best detected bounding box for the element.
        alternative_boundaries: Other possible bounding boxes.
        detection_strategies_used: Strategies that were attempted.
        pixel_data: Extracted pixel data (RGB).
        mask: Optional mask for non-rectangular elements.
        application_hint: Optional application name for profile lookup.
        created_at: When this candidate was created.
        confidence_score: Overall confidence in the detection (0-1).
        element_type: Detected type of the element.
    """

    # Identity
    id: str
    session_id: str

    # Click context
    click_x: int
    click_y: int
    click_button: str
    timestamp: float
    frame_number: int

    # Detection results
    primary_boundary: InferredBoundingBox
    alternative_boundaries: list[InferredBoundingBox] = field(default_factory=list)
    detection_strategies_used: list[DetectionStrategy] = field(default_factory=list)

    # Template data (stored as numpy arrays)
    pixel_data: np.ndarray | None = None
    mask: np.ndarray | None = None

    # Metadata
    application_hint: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.0
    element_type: str = "unknown"

    @property
    def pixel_data_base64(self) -> str | None:
        """Get pixel data encoded as base64 for transport."""
        return _encode_array(self.pixel_data)

    @property
    def mask_base64(self) -> str | None:
        """Get mask encoded as base64 for transport."""
        return _encode_array(self.mask)

    @property
    def width(self) -> int:
        """Width of the detected element."""
        return self.primary_boundary.width

    @property
    def height(self) -> int:
        """Height of the detected element."""
        return self.primary_boundary.height

    @property
    def bounding_box(self) -> tuple[int, int, int, int]:
        """Return (x, y, width, height) tuple."""
        return (
            self.primary_boundary.x,
            self.primary_boundary.y,
            self.primary_boundary.width,
            self.primary_boundary.height,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "click_x": self.click_x,
            "click_y": self.click_y,
            "click_button": self.click_button,
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "primary_boundary": self.primary_boundary.to_dict(),
            "alternative_boundaries": [b.to_dict() for b in self.alternative_boundaries],
            "detection_strategies_used": [s.value for s in self.detection_strategies_used],
            "pixel_data_base64": self.pixel_data_base64,
            "mask_base64": self.mask_base64,
            "pixel_shape": list(self.pixel_data.shape) if self.pixel_data is not None else None,
            "mask_shape": list(self.mask.shape) if self.mask is not None else None,
            "application_hint": self.application_hint,
            "created_at": self.created_at.isoformat(),
            "confidence_score": self.confidence_score,
            "element_type": self.element_type,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClickTemplateCandidate":
        """Create from dictionary."""
        from .models import DetectionStrategy, ElementType, InferredBoundingBox

        # Reconstruct primary boundary
        pb = data["primary_boundary"]
        primary_boundary = InferredBoundingBox(
            x=pb["x"],
            y=pb["y"],
            width=pb["width"],
            height=pb["height"],
            confidence=pb["confidence"],
            strategy_used=DetectionStrategy(pb["strategy_used"]),
            element_type=ElementType(pb.get("element_type", "unknown")),
        )

        # Reconstruct alternative boundaries
        alt_boundaries = []
        for ab in data.get("alternative_boundaries", []):
            alt_boundaries.append(
                InferredBoundingBox(
                    x=ab["x"],
                    y=ab["y"],
                    width=ab["width"],
                    height=ab["height"],
                    confidence=ab["confidence"],
                    strategy_used=DetectionStrategy(ab["strategy_used"]),
                    element_type=ElementType(ab.get("element_type", "unknown")),
                )
            )

        # Decode pixel data if present
        pixel_data = None
        if data.get("pixel_data_base64") and data.get("pixel_shape"):
            pixel_data = _decode_array(
                data["pixel_data_base64"],
                tuple(data["pixel_shape"]),
                np.dtype(np.uint8),
            )

        # Decode mask if present
        mask = None
        if data.get("mask_base64") and data.get("mask_shape"):
            mask = _decode_array(
                data["mask_base64"],
                tuple(data["mask_shape"]),
                np.dtype(np.float32),
            )

        return cls(
            id=data["id"],
            session_id=data["session_id"],
            click_x=data["click_x"],
            click_y=data["click_y"],
            click_button=data["click_button"],
            timestamp=data["timestamp"],
            frame_number=data["frame_number"],
            primary_boundary=primary_boundary,
            alternative_boundaries=alt_boundaries,
            detection_strategies_used=[
                DetectionStrategy(s) for s in data.get("detection_strategies_used", [])
            ],
            pixel_data=pixel_data,
            mask=mask,
            application_hint=data.get("application_hint"),
            created_at=datetime.fromisoformat(data["created_at"]),
            confidence_score=data.get("confidence_score", 0.0),
            element_type=data.get("element_type", "unknown"),
        )
