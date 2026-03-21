"""Base classes for the cascade detection backend system.

Defines the DetectionBackend ABC and DetectionResult dataclass that all
detection backends must implement. Backends are ordered by estimated cost
and tried in sequence by the CascadeDetector.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectionResult:
    """Result from a detection backend.

    Attributes:
        x: X coordinate of the detected element (top-left).
        y: Y coordinate of the detected element (top-left).
        width: Width of the detected region.
        height: Height of the detected region.
        confidence: Confidence score (0.0-1.0).
        backend_name: Name of the backend that produced this result.
        metadata: Optional backend-specific metadata.
    """

    x: int
    y: int
    width: int
    height: int
    confidence: float
    backend_name: str
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        """Center point of the detected region."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Bounding box as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)


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
