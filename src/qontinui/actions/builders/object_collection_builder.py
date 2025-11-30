"""Unified ObjectCollectionBuilder composed from focused sub-builders.

Delegates to specialized builders for different object types.
"""

from typing import TYPE_CHECKING, Optional

from .image_collection_builder import ImageCollectionBuilder
from .match_collection_builder import MatchCollectionBuilder
from .region_collection_builder import RegionCollectionBuilder
from .scene_collection_builder import SceneCollectionBuilder
from .string_collection_builder import StringCollectionBuilder

if TYPE_CHECKING:
    from ...model.match import Match
    from ...model.state import State, StateImage, StateLocation, StateRegion, StateString
    from ..action_result import ActionResult
    from ..object_collection import ObjectCollection


class ObjectCollectionBuilder:
    """Builder for creating ObjectCollection instances fluently.

    Composed from focused sub-builders:
    - ImageCollectionBuilder: StateImages, Patterns
    - RegionCollectionBuilder: Regions, Locations
    - MatchCollectionBuilder: Matches, ActionResults
    - StringCollectionBuilder: StateStrings, text
    - SceneCollectionBuilder: Scenes

    Uses builder composition pattern for separation of concerns.
    """

    def __init__(self) -> None:
        """Initialize builder with sub-builders."""
        self.images = ImageCollectionBuilder()
        self.regions = RegionCollectionBuilder()
        self.matches = MatchCollectionBuilder()
        self.strings = StringCollectionBuilder()
        self.scenes = SceneCollectionBuilder()

    # Image-related methods

    def with_images(self, *state_images) -> "ObjectCollectionBuilder":
        """Add state images to collection.

        Args:
            state_images: Variable number of StateImage objects or list

        Returns:
            This builder for method chaining
        """
        self.images.with_images(*state_images)
        return self

    def set_images(self, state_images: list["StateImage"]) -> "ObjectCollectionBuilder":
        """Set state images list.

        Args:
            state_images: List of StateImage objects

        Returns:
            This builder for method chaining
        """
        self.images.set_images(state_images)
        return self

    def with_patterns(self, *patterns) -> "ObjectCollectionBuilder":
        """Add patterns as state images.

        Args:
            patterns: Variable number of Pattern objects or list

        Returns:
            This builder for method chaining
        """
        self.images.with_patterns(*patterns)
        return self

    def with_all_state_images(self, state: Optional["State"]) -> "ObjectCollectionBuilder":
        """Add all state images from a state.

        Args:
            state: State to get images from

        Returns:
            This builder for method chaining
        """
        self.images.with_all_state_images(state)
        return self

    def with_non_shared_images(self, state: Optional["State"]) -> "ObjectCollectionBuilder":
        """Add non-shared state images from a state.

        Args:
            state: State to get images from

        Returns:
            This builder for method chaining
        """
        self.images.with_non_shared_images(state)
        return self

    # Region-related methods

    def with_locations(self, *locations) -> "ObjectCollectionBuilder":
        """Add locations to collection.

        Args:
            locations: Variable number of Location or StateLocation objects

        Returns:
            This builder for method chaining
        """
        self.regions.with_locations(*locations)
        return self

    def set_locations(self, locations: list["StateLocation"]) -> "ObjectCollectionBuilder":
        """Set locations list.

        Args:
            locations: List of StateLocation objects

        Returns:
            This builder for method chaining
        """
        self.regions.set_locations(locations)
        return self

    def with_regions(self, *regions) -> "ObjectCollectionBuilder":
        """Add regions to collection.

        Args:
            regions: Variable number of Region or StateRegion objects

        Returns:
            This builder for method chaining
        """
        self.regions.with_regions(*regions)
        return self

    def set_regions(self, regions: list["StateRegion"]) -> "ObjectCollectionBuilder":
        """Set regions list.

        Args:
            regions: List of StateRegion objects

        Returns:
            This builder for method chaining
        """
        self.regions.set_regions(regions)
        return self

    def with_grid_subregions(self, rows: int, columns: int, *regions) -> "ObjectCollectionBuilder":
        """Add grid subregions from regions.

        Args:
            rows: Number of rows in grid
            columns: Number of columns in grid
            regions: Variable number of Region or StateRegion objects

        Returns:
            This builder for method chaining
        """
        self.regions.with_grid_subregions(rows, columns, *regions)
        return self

    # Match-related methods

    def with_matches(self, *matches: "ActionResult") -> "ObjectCollectionBuilder":
        """Add matches to collection.

        Args:
            matches: Variable number of ActionResult objects

        Returns:
            This builder for method chaining
        """
        self.matches.with_matches(*matches)
        return self

    def set_matches(self, matches: list["ActionResult"]) -> "ObjectCollectionBuilder":
        """Set matches list.

        Args:
            matches: List of ActionResult objects

        Returns:
            This builder for method chaining
        """
        self.matches.set_matches(matches)
        return self

    def with_match_objects_as_regions(self, *matches: "Match") -> "ObjectCollectionBuilder":
        """Add match objects as regions.

        Args:
            matches: Variable number of Match objects

        Returns:
            This builder for method chaining
        """
        self.matches.with_match_objects_as_regions(*matches)
        return self

    def with_match_objects_as_state_images(self, *matches: "Match") -> "ObjectCollectionBuilder":
        """Add match objects as state images.

        Args:
            matches: Variable number of Match objects

        Returns:
            This builder for method chaining
        """
        self.matches.with_match_objects_as_state_images(*matches)
        return self

    # String-related methods

    def with_strings(self, *strings) -> "ObjectCollectionBuilder":
        """Add strings to collection.

        Args:
            strings: Variable number of string or StateString objects

        Returns:
            This builder for method chaining
        """
        self.strings.with_strings(*strings)
        return self

    def set_strings(self, strings: list["StateString"]) -> "ObjectCollectionBuilder":
        """Set strings list.

        Args:
            strings: List of StateString objects

        Returns:
            This builder for method chaining
        """
        self.strings.set_strings(strings)
        return self

    # Scene-related methods

    def with_scenes(self, *scenes) -> "ObjectCollectionBuilder":
        """Add scenes to collection.

        Args:
            scenes: Variable number of string, Pattern, Scene objects or list

        Returns:
            This builder for method chaining
        """
        self.scenes.with_scenes(*scenes)
        return self

    # Build method

    def build(self) -> "ObjectCollection":
        """Create the ObjectCollection with configured properties.

        Combines all sub-builders into a unified ObjectCollection.

        Returns:
            Configured ObjectCollection instance
        """
        from ..object_collection import ObjectCollection

        # Build from sub-builders
        state_images = self.images.build()
        state_locations, state_regions = self.regions.build()
        matches, match_regions, match_images = self.matches.build()
        state_strings = self.strings.build()
        scenes = self.scenes.build()

        # Combine match-derived objects
        state_regions.extend(match_regions)
        state_images.extend(match_images)

        # Create and populate ObjectCollection
        return ObjectCollection(
            state_locations=state_locations,
            state_images=state_images,
            state_regions=state_regions,
            state_strings=state_strings,
            matches=matches,
            scenes=scenes,
        )
