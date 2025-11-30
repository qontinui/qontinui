"""StateImage - ported from Qontinui framework.

Images associated with states for identification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...find import Matches
from ..element import Image, Pattern, Region
from ..search_regions import SearchRegions
from .action_history import ActionHistory

if TYPE_CHECKING:
    from qontinui.model.state.state import State


@dataclass
class StateImage:
    """Image associated with a state.

    Port of StateImage from Qontinui framework class.
    Represents an image that helps identify a state.
    """

    image: Image | Pattern
    name: str | None = None
    owner_state: State | None = None

    # Image properties
    _fixed: bool = False  # If true, always appears in same location
    _shared: bool = False  # If true, can appear in multiple states
    _probability: float = 1.0  # Probability this image appears in state

    # Search configuration
    _search_region: Region | None = None
    _search_regions: SearchRegions | None = None  # SearchRegions associated with this StateImage

    # Similarity threshold for this state image
    #
    # Similarity Priority Cascade (highest to lowest):
    # 1. FindOptions.similarity (action-level, explicit) - HIGHEST
    # 2. Pattern.similarity (image-level override)
    # 3. StateImage._similarity (THIS LEVEL - from JSON config)
    # 4. QontinuiSettings.similarity_threshold (project config = 0.85)
    # 5. Library default (action_defaults = 0.7) - LOWEST
    #
    # This value is applied to Patterns created from this StateImage (via get_pattern())
    # only if the Pattern doesn't already have its own similarity set
    _similarity: float = 0.7

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Action history for integration testing
    action_history: ActionHistory = field(default_factory=ActionHistory)

    # Screenshot IDs for pixel stability analysis
    screenshot_ids: list[str] = field(default_factory=list)

    @property
    def screenshots(self) -> list[str]:
        """Get screenshot IDs as property.

        Returns:
            List of screenshot IDs
        """
        return self.screenshot_ids

    def __post_init__(self):
        """Initialize StateImage properties."""
        if self.name is None:
            if isinstance(self.image, Pattern):
                self.name = self.image.name
            elif isinstance(self.image, Image):
                self.name = self.image.name

    def get_pattern(self) -> Pattern:
        """Get pattern for finding with proper similarity cascade.

        Similarity Priority:
        1. Pattern.similarity (if already set) - respects pattern-level override
        2. StateImage._similarity (this level) - applies if pattern has no similarity
        3. Lower priorities handled by FindOptions

        Returns:
            Pattern object with appropriate configuration
        """
        if isinstance(self.image, Pattern):
            pattern = self.image
        else:
            pattern = Pattern.from_image(self.image)

        # Apply StateImage similarity only if Pattern doesn't have its own
        # This respects the similarity priority hierarchy
        if pattern.similarity is None:
            pattern = pattern.with_similarity(self._similarity)

        # Always apply search region from StateImage
        if self._search_region:
            pattern = pattern.with_search_region(self._search_region)

        return pattern

    def find(self) -> Matches:
        """Find this state image using FindAction with proper cascade.

        Returns:
            Matches object with all found matches
        """
        from ...actions.find import FindAction
        from ...actions.find.find_options_builder import CascadeContext, build_find_options

        pattern = self.get_pattern()
        action = FindAction()

        # Build options with full cascade
        try:
            from ...config.settings import QontinuiSettings

            project_config = QontinuiSettings()
        except Exception:
            project_config = None

        ctx = CascadeContext(
            search_options=None,  # StateImage doesn't have SearchOptions
            pattern=pattern,
            state_image=self,
            project_config=project_config,
        )
        options = build_find_options(ctx, explicit_find_all=True)

        result = action.find(pattern=pattern, options=options)
        return Matches(result.matches)  # type: ignore[arg-type]

    def exists(self) -> bool:
        """Check if this state image exists using FindAction with proper cascade.

        Returns:
            True if image found on screen
        """
        from ...actions.find import FindAction
        from ...actions.find.find_options_builder import CascadeContext, build_find_options

        pattern = self.get_pattern()
        action = FindAction()

        # Build options with full cascade
        try:
            from ...config.settings import QontinuiSettings

            project_config = QontinuiSettings()
        except Exception:
            project_config = None

        ctx = CascadeContext(
            search_options=None,  # StateImage doesn't have SearchOptions
            pattern=pattern,
            state_image=self,
            project_config=project_config,
        )
        options = build_find_options(ctx)

        result = action.find(pattern=pattern, options=options)
        return result.found

    def wait_for(self, timeout: float = 5.0) -> bool:
        """Wait for image to appear with proper cascade.

        Args:
            timeout: Maximum wait time (default: 5.0 seconds)

        Returns:
            True if image appeared
        """
        from ...actions.find import FindAction
        from ...actions.find.find_options_builder import CascadeContext, build_find_options

        pattern = self.get_pattern()
        action = FindAction()

        # Build options with full cascade
        try:
            from ...config.settings import QontinuiSettings

            project_config = QontinuiSettings()
        except Exception:
            project_config = None

        ctx = CascadeContext(
            search_options=None,
            pattern=pattern,
            state_image=self,
            project_config=project_config,
        )
        options = build_find_options(ctx, explicit_timeout=timeout)

        result = action.find(pattern=pattern, options=options)
        return result.found

    def set_fixed(self, fixed: bool = True) -> StateImage:
        """Set whether image is fixed in position (fluent).

        Args:
            fixed: True if image is fixed

        Returns:
            Self for chaining
        """
        self._fixed = fixed
        return self

    def set_shared(self, shared: bool = True) -> StateImage:
        """Set whether image is shared across states (fluent).

        Args:
            shared: True if image is shared

        Returns:
            Self for chaining
        """
        self._shared = shared
        return self

    def set_probability(self, probability: float) -> StateImage:
        """Set probability that image appears (fluent).

        Args:
            probability: Probability (0.0 to 1.0)

        Returns:
            Self for chaining
        """
        self._probability = max(0.0, min(1.0, probability))
        return self

    def set_search_region(self, region: Region) -> StateImage:
        """Set search region (fluent).

        Args:
            region: Region to search in

        Returns:
            Self for chaining
        """
        self._search_region = region
        return self

    def set_similarity(self, similarity: float) -> StateImage:
        """Set similarity threshold (fluent).

        Args:
            similarity: Similarity (0.0 to 1.0)

        Returns:
            Self for chaining
        """
        self._similarity = max(0.0, min(1.0, similarity))
        return self

    def set_search_regions(self, search_regions: SearchRegions) -> StateImage:
        """Set search regions for this image (fluent).

        Args:
            search_regions: SearchRegions to use for finding this image

        Returns:
            Self for chaining
        """
        self._search_regions = search_regions
        return self

    @property
    def search_regions(self) -> SearchRegions | None:
        """Get search regions for this image."""
        return self._search_regions

    @property
    def is_fixed(self) -> bool:
        """Check if image is fixed in position."""
        return self._fixed

    @property
    def is_shared(self) -> bool:
        """Check if image is shared across states."""
        return self._shared

    @property
    def probability(self) -> float:
        """Get probability that image appears."""
        return self._probability

    @property
    def patterns(self) -> list[Pattern]:
        """Get patterns for this state image.

        Returns:
            List containing the pattern
        """
        pattern = self.get_pattern()
        return [pattern]

    @property
    def owner_state_name(self) -> str:
        """Get owner state name as a property.

        Returns:
            Owner state name or empty string
        """
        return self.owner_state.name if self.owner_state else ""

    def get_name(self) -> str:
        """Get the name of this state image.

        Returns:
            Name or empty string
        """
        return self.name or ""

    def get_patterns(self) -> list[Pattern]:
        """Get patterns for finding this image.

        Returns:
            List containing the pattern
        """
        return self.patterns

    def get_owner_state_name(self) -> str:
        """Get the owner state name.

        Returns:
            Owner state name or empty string
        """
        return self.owner_state_name

    def set_times_acted_on(self, times: int) -> None:
        """Set times this image has been acted upon.

        Args:
            times: Number of times acted upon
        """
        # This would typically update action history
        # For now, just a placeholder
        pass

    def get_all_match_snapshots(self) -> list[Any]:
        """Get all match snapshots for this state image.

        Returns:
            List of match snapshots
        """
        # This would typically return historical match data
        # For now, return empty list
        return []

    def __str__(self) -> str:
        """String representation."""
        state_name = self.owner_state.name if self.owner_state else "None"
        return f"StateImage('{self.name}' in state '{state_name}')"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"StateImage(name='{self.name}', fixed={self._fixed}, shared={self._shared})"
