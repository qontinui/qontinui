"""Data types for action caching.

Provides type definitions for cached action entries and results.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CachedCoordinates:
    """Cached coordinates for an action target."""

    x: int
    """X coordinate of element center."""

    y: int
    """Y coordinate of element center."""

    region_x: int
    """X coordinate of element bounding box top-left."""

    region_y: int
    """Y coordinate of element bounding box top-left."""

    region_width: int
    """Width of element bounding box."""

    region_height: int
    """Height of element bounding box."""


@dataclass
class ValidationPattern:
    """Pattern data for validating cached entries."""

    content_hash: str
    """Hash of the pattern pixel data."""

    width: int
    """Width of the pattern."""

    height: int
    """Height of the pattern."""

    sample_region: tuple[int, int, int, int] | None = None
    """Optional (x, y, w, h) region to sample for quick validation."""

    sample_hash: str | None = None
    """Hash of the sample region for quick validation."""


@dataclass
class CacheEntry:
    """A cached action entry."""

    coordinates: CachedCoordinates
    """Cached coordinates where the element was found."""

    confidence: float
    """Match confidence when this entry was created (0.0-1.0)."""

    timestamp: float
    """Unix timestamp when this entry was created."""

    validation_pattern: ValidationPattern | None = None
    """Optional pattern for validating the cache entry is still valid."""

    hit_count: int = 0
    """Number of times this entry has been used."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the cached action."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "coordinates": {
                "x": self.coordinates.x,
                "y": self.coordinates.y,
                "region_x": self.coordinates.region_x,
                "region_y": self.coordinates.region_y,
                "region_width": self.coordinates.region_width,
                "region_height": self.coordinates.region_height,
            },
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "validation_pattern": (
                {
                    "content_hash": self.validation_pattern.content_hash,
                    "width": self.validation_pattern.width,
                    "height": self.validation_pattern.height,
                    "sample_region": self.validation_pattern.sample_region,
                    "sample_hash": self.validation_pattern.sample_hash,
                }
                if self.validation_pattern
                else None
            ),
            "hit_count": self.hit_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        """Create from dictionary (JSON deserialization)."""
        coords_data = data["coordinates"]
        coordinates = CachedCoordinates(
            x=coords_data["x"],
            y=coords_data["y"],
            region_x=coords_data["region_x"],
            region_y=coords_data["region_y"],
            region_width=coords_data["region_width"],
            region_height=coords_data["region_height"],
        )

        validation_pattern = None
        if data.get("validation_pattern"):
            vp_data = data["validation_pattern"]
            validation_pattern = ValidationPattern(
                content_hash=vp_data["content_hash"],
                width=vp_data["width"],
                height=vp_data["height"],
                sample_region=(
                    tuple(vp_data["sample_region"]) if vp_data.get("sample_region") else None
                ),
                sample_hash=vp_data.get("sample_hash"),
            )

        return cls(
            coordinates=coordinates,
            confidence=data["confidence"],
            timestamp=data["timestamp"],
            validation_pattern=validation_pattern,
            hit_count=data.get("hit_count", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CacheResult:
    """Result of a cache lookup."""

    hit: bool
    """Whether the cache lookup was a hit."""

    entry: CacheEntry | None = None
    """The cached entry if hit=True."""

    invalidation_reason: str | None = None
    """Reason for cache miss/invalidation if hit=False."""


@dataclass
class CacheStats:
    """Statistics about cache usage."""

    total_entries: int
    """Total number of entries in cache."""

    hits: int
    """Total cache hits since startup."""

    misses: int
    """Total cache misses since startup."""

    invalidations: int
    """Total cache invalidations since startup."""

    size_bytes: int
    """Approximate size of cache in bytes."""

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate as percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
