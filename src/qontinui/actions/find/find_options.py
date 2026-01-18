"""FindOptions - Configuration for find operations.

This module handles options for FIND, EXISTS, and other image search actions.
Options can come from multiple sources with a defined priority cascade.

IMPORTANT: This is a simplified FindOptions for Python-native actions.
For JSON config actions (using SearchOptions from frontend), additional
conversion logic is needed to map all SearchOptions fields to FindOptions
or directly to the underlying find implementation.

TODO: Expand this class or create conversion utility to support:
- search_regions (list of regions)
- max_matches/min_matches
- polling configuration
- pattern options (scale_invariant, rotation_invariant, etc.)
- match adjustments
- capture_image flag

These would all follow the same cascade priority as similarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...model.element import Region


def _get_default_similarity() -> float:
    """Get default similarity using proper cascade priority.

    Priority (highest to lowest):
    1. Pattern.similarity (if set on the pattern being searched)
    2. Project config (QontinuiSettings.similarity_threshold)
    3. Library default (action_defaults.FindActionDefaults.default_similarity_threshold)

    Note: This function provides the baseline. Pattern-level similarity
    is handled separately when Pattern is passed to find operations.
    """
    # Try to get project-level config first
    try:
        from ...config.settings import QontinuiSettings

        settings = QontinuiSettings()
        return settings.similarity_threshold
    except Exception:
        # Fall back to library default if project config unavailable
        pass

    # Library default (lowest priority)
    from ...config.action_defaults import get_defaults

    return get_defaults().find.default_similarity_threshold


@dataclass
class FindOptions:
    """Options for find operations.

    Used by all find operations regardless of mock/real execution.

    Similarity Priority:
    1. FindOptions.similarity (if explicitly set) - highest priority
    2. Pattern.similarity (if set on pattern)
    3. StateImage._similarity (if using StateImage)
    4. Project config (QontinuiSettings.similarity_threshold)
    5. Library default (action_defaults) - lowest priority

    Image Variant Options:
    - grayscale: Convert to grayscale before matching
    - edge_detection: Use edge detection for matching
    - scale_invariant: Enable scale-invariant matching (SIFT/SURF)
    - rotation_invariant: Enable rotation-invariant matching
    - color_tolerance: Tolerance for color differences (0-255)
    """

    # Core options
    similarity: float = field(default_factory=_get_default_similarity)
    find_all: bool = False
    search_region: Region | None = None
    timeout: float = 0.0
    collect_debug: bool = False
    monitor_index: int | None = None  # Target monitor for screenshot capture (None = all monitors)

    # Image variant options (from FindImage)
    grayscale: bool = False
    """Convert images to grayscale before matching."""

    edge_detection: bool = False
    """Apply edge detection (Canny) before matching."""

    scale_invariant: bool = False
    """Use scale-invariant feature matching (SIFT/ORB)."""

    rotation_invariant: bool = False
    """Use rotation-invariant feature matching."""

    color_tolerance: int = 0
    """Tolerance for color differences (0-255). 0 means exact match."""

    # Self-healing options
    enable_healing: bool = False
    """Enable self-healing when pattern matching fails."""

    healing_context_description: str = ""
    """Description of element for healing context (e.g., 'login button')."""

    state_id: str | None = None
    """Current state machine state ID for cache key generation."""

    action_type: str | None = None
    """Type of action (click, type, etc.) for cache key and healing context."""

    # Caching options
    use_cache: bool = False
    """Check action cache before template matching."""

    store_in_cache: bool = False
    """Store successful matches in action cache."""

    # Validation options
    enable_validation: bool = False
    """Enable visual validation after successful match."""

    pre_screenshot: Any = None
    """Pre-action screenshot for validation (captured before action). numpy.ndarray or None."""
