"""Base classes for the cascade detection backend system.

Defines the DetectionBackend ABC and DetectionResult dataclass that all
detection backends must implement. Backends are ordered by estimated cost
and tried in sequence by the CascadeDetector.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass
class DetectionResult:
    """Result from a detection backend.

    Attributes:
        x: X coordinate of the detected element (top-left), in pixels.
        y: Y coordinate of the detected element (top-left), in pixels.
        width: Width of the detected region, in pixels.
        height: Height of the detected region, in pixels.
        confidence: Confidence score (0.0-1.0).
        backend_name: Name of the backend that produced this result.
        label: Optional label describing the detected element.
        metadata: Optional backend-specific metadata.
        normalized_x: X coordinate normalized to 0.0-1.0 range (screen-relative).
        normalized_y: Y coordinate normalized to 0.0-1.0 range (screen-relative).
        normalized_width: Width normalized to 0.0-1.0 range (screen-relative).
        normalized_height: Height normalized to 0.0-1.0 range (screen-relative).
    """

    x: int
    y: int
    width: int
    height: int
    confidence: float
    backend_name: str
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    normalized_x: float | None = None
    normalized_y: float | None = None
    normalized_width: float | None = None
    normalized_height: float | None = None

    @property
    def center(self) -> tuple[int, int]:
        """Center point of the detected region."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Bounding box as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    @property
    def normalized_bounds(self) -> tuple[float, float | None, float | None, float | None] | None:
        """Normalized bounding box as (x, y, width, height) in 0.0-1.0 range.

        Returns None if coordinates have not been normalized yet.
        """
        if self.normalized_x is None:
            return None
        return (
            self.normalized_x,
            self.normalized_y,
            self.normalized_width,
            self.normalized_height,
        )

    def normalize(self, screen_width: int, screen_height: int) -> "DetectionResult":
        """Return a copy with normalized coordinates filled in.

        Normalized coordinates represent position and size as fractions of
        the screen dimensions (0.0-1.0), making them resolution-independent.

        Args:
            screen_width: Full screen width in pixels.
            screen_height: Full screen height in pixels.

        Returns:
            A new DetectionResult with normalized_* fields populated.
        """
        if screen_width <= 0 or screen_height <= 0:
            return self
        return replace(
            self,
            normalized_x=self.x / screen_width,
            normalized_y=self.y / screen_height,
            normalized_width=self.width / screen_width,
            normalized_height=self.height / screen_height,
        )


class DetectionBackend(ABC):
    """Abstract base class for detection backends.

    Each backend wraps an existing detection mechanism (template matching,
    feature matching, OCR, accessibility, vision LLM) behind a uniform
    interface. Backends report their estimated cost so the CascadeDetector
    can order them cheapest-first.
    """

    @abstractmethod
    def find(self, needle: Any, haystack: Any, config: dict[str, Any]) -> list[DetectionResult]:
        """Find needle in haystack.

        Args:
            needle: What to search for (image, text, selector, etc.).
            haystack: Where to search (screenshot image).
            config: Detection configuration with keys like:
                - needle_type: str (template, text, accessibility_id, etc.)
                - min_confidence: float
                - find_all: bool
                - search_region: tuple[int, int, int, int] | None

        Returns:
            List of DetectionResult sorted by confidence (highest first).
            Empty list on failure.
        """
        ...

    @abstractmethod
    def supports(self, needle_type: str) -> bool:
        """Whether this backend can handle the given needle type.

        Args:
            needle_type: Type of needle (template, text, accessibility_id,
                        role, label, description).

        Returns:
            True if this backend supports the needle type.
        """
        ...

    @abstractmethod
    def estimated_cost_ms(self) -> float:
        """Estimated latency in milliseconds.

        Used by CascadeDetector to order backends cheapest-first.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this backend."""
        ...

    def is_available(self) -> bool:
        """Whether this backend is currently usable.

        Override to check for model availability, service reachability, etc.
        Returns True by default.
        """
        return True

    def to_detections(
        self,
        results: list[DetectionResult],
        *,
        screen_width: int = 0,
        screen_height: int = 0,
    ):
        """Convert results from this backend into a Detections container.

        Uses lazy import to avoid circular dependency with detections module.

        Args:
            results: DetectionResult list from this backend's ``find()``.
            screen_width: Screen width for normalization (0 to skip).
            screen_height: Screen height for normalization (0 to skip).

        Returns:
            A ``Detections`` container.
        """
        from ..detections import Detections

        return Detections.from_detection_results(
            results,
            screen_width=screen_width,
            screen_height=screen_height,
        )
