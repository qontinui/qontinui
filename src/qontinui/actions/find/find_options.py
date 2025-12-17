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

from dataclasses import dataclass, field

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
    """

    similarity: float = field(default_factory=_get_default_similarity)
    find_all: bool = False
    search_region: Region | None = None
    timeout: float = 0.0
    collect_debug: bool = False
    monitor_index: int | None = (
        None  # Target monitor for screenshot capture (None = all monitors)
    )
