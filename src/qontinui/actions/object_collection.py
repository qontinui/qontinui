"""Object collection - ported from Qontinui framework.

Container for GUI elements serving as action targets.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..model.element import Location, Pattern, Region, Scene
    from ..model.element.position import Position
    from ..model.element.positions import Name as PositionName
    from ..model.match import Match
    from ..model.state import State, StateImage, StateLocation, StateRegion, StateString
    from .action_result import ActionResult


@dataclass
class ObjectCollection:
    """Container for GUI elements that serve as targets for automation actions.

    Port of ObjectCollection from Qontinui framework class.

    ObjectCollection is a fundamental data structure in the model-based approach that
    aggregates different types of GUI elements that can be acted upon. It provides a
    unified way to pass multiple heterogeneous targets to actions, supporting the
    framework's flexibility in handling various GUI interaction scenarios.

    Supported element types:
    - StateImages: Visual patterns to find and interact with
    - StateLocations: Specific points for precise interactions
    - StateRegions: Rectangular areas for spatial operations
    - StateStrings: Text strings for keyboard input
    - ActionResult: Results from previous find operations that can be reused
    - Scenes: Screenshots for offline processing instead of live screen capture

    Key features:
    - Supports multiple objects of each type for batch operations
    - Tracks interaction counts for each object (times_acted_on)
    - Enables offline automation using stored screenshots (Scenes)
    - Provides builder pattern for convenient construction

    This design allows actions to be polymorphic - the same action (e.g., Click) can
    operate on images, regions, locations, or previous matches, making automation code
    more flexible and reusable.
    """

    state_locations: list["StateLocation"] = field(default_factory=list)
    """List of state locations."""

    state_images: list["StateImage"] = field(default_factory=list)
    """List of state images."""

    state_regions: list["StateRegion"] = field(default_factory=list)
    """List of state regions."""

    state_strings: list["StateString"] = field(default_factory=list)
    """List of state strings."""

    matches: list["ActionResult"] = field(default_factory=list)
    """List of action results."""

    scenes: list["Scene"] = field(default_factory=list)
    """List of scenes."""

    def is_empty(self) -> bool:
        """Check if collection is empty.

        Returns:
            True if all lists are empty
        """
        return (
            len(self.state_locations) == 0
            and len(self.state_images) == 0
            and len(self.state_regions) == 0
            and len(self.state_strings) == 0
            and len(self.matches) == 0
            and len(self.scenes) == 0
        )

    def reset_times_acted_on(self) -> None:
        """Reset times_acted_on for all objects.

        Knowing how many times an object Match was acted on is valuable
        for understanding the actual automation as well as for performing mocks.
        """
        for sio in self.state_images:
            sio.set_times_acted_on(0)
        for sio in self.state_locations:
            sio.set_times_acted_on(0)
        for sio in self.state_regions:
            sio.set_times_acted_on(0)
        for sio in self.state_strings:
            sio.set_times_acted_on(0)
        for m in self.matches:
            m.set_times_acted_on(0)

    def get_first_object_name(self) -> str:
        """Get name of first object in collection.

        Returns:
            Name of first object or empty string
        """
        if self.state_images:
            if self.state_images[0].get_name():
                return self.state_images[0].get_name()
            elif (
                self.state_images[0].get_patterns()
                and self.state_images[0].get_patterns()[0].get_imgpath()
            ):
                return self.state_images[0].get_patterns()[0].get_imgpath()
        if self.state_locations and self.state_locations[0].get_name():
            return self.state_locations[0].get_name()
        if self.state_regions and self.state_regions[0].get_name():
            return self.state_regions[0].get_name()
        if self.state_strings:
            return self.state_strings[0].get_string()
        return ""

    def contains(self, obj) -> bool:
        """Check if collection contains an object.

        Args:
            obj: Object to check for

        Returns:
            True if object is in collection
        """
        if isinstance(obj, StateImage):
            for si in self.state_images:
                if obj == si:
                    return True
            return False
        elif isinstance(obj, StateRegion):
            return obj in self.state_regions
        elif isinstance(obj, StateLocation):
            return obj in self.state_locations
        elif isinstance(obj, StateString):
            return obj in self.state_strings
        elif isinstance(obj, ActionResult):
            return obj in self.matches
        elif isinstance(obj, Scene):
            return obj in self.scenes
        return False

    def equals(self, object_collection: "ObjectCollection") -> bool:
        """Check equality with another ObjectCollection.

        Args:
            object_collection: Collection to compare with

        Returns:
            True if collections contain same objects
        """
        for si in self.state_images:
            if not object_collection.contains(si):
                return False
        for sr in self.state_regions:
            if not object_collection.contains(sr):
                return False
        for sl in self.state_locations:
            if not object_collection.contains(sl):
                return False
        for ss in self.state_strings:
            if not object_collection.contains(ss):
                return False
        for m in self.matches:
            if not object_collection.contains(m):
                return False
        for sc in self.scenes:
            if not object_collection.contains(sc):
                return False
        return True

    def get_all_image_filenames(self) -> set[str]:
        """Get all image filenames from state images.

        Returns:
            Set of unique image filenames
        """
        filenames = set()
        for si in self.state_images:
            for p in si.get_patterns():
                filenames.add(p.get_imgpath())
        return filenames

    def get_all_owner_states(self) -> set[str]:
        """Get all owner state names.

        Returns:
            Set of unique owner state names
        """
        states = set()
        for si in self.state_images:
            states.add(si.get_owner_state_name())
        for si in self.state_locations:
            states.add(si.get_owner_state_name())
        for si in self.state_regions:
            states.add(si.get_owner_state_name())
        for si in self.state_strings:
            states.add(si.get_owner_state_name())
        return states

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"ObjectCollection{{stateLocations={len(self.state_locations)}, "
            f"stateImages={len(self.state_images)}, "
            f"stateRegions={len(self.state_regions)}, "
            f"stateStrings={len(self.state_strings)}, "
            f"matches={len(self.matches)}, "
            f"scenes={len(self.scenes)}}}"
        )


class ObjectCollectionBuilder:
    """Builder for creating ObjectCollection instances fluently.

    Port of ObjectCollection from Qontinui framework.Builder class.
    """

    def __init__(self):
        """Initialize builder with empty lists."""
        self.state_locations: list[StateLocation] = []
        self.state_images: list[StateImage] = []
        self.state_regions: list[StateRegion] = []
        self.state_strings: list[StateString] = []
        self.matches: list[ActionResult] = []
        self.scenes: list[Scene] = []

    def with_locations(self, *locations) -> "ObjectCollectionBuilder":
        """Add locations to collection.

        Args:
            locations: Variable number of Location or StateLocation objects

        Returns:
            This builder for method chaining
        """
        for location in locations:
            if isinstance(location, Location):
                state_location = location.as_state_location_in_null_state()
                state_location.set_position(Position(PositionName.TOPLEFT))
                self.state_locations.append(state_location)
            elif isinstance(location, StateLocation):
                self.state_locations.append(location)
        return self

    def set_locations(self, locations: list["StateLocation"]) -> "ObjectCollectionBuilder":
        """Set locations list.

        Args:
            locations: List of StateLocation objects

        Returns:
            This builder for method chaining
        """
        self.state_locations = locations
        return self

    def with_images(self, *state_images) -> "ObjectCollectionBuilder":
        """Add state images to collection.

        Args:
            state_images: Variable number of StateImage objects or list

        Returns:
            This builder for method chaining
        """
        for item in state_images:
            if isinstance(item, list):
                self.state_images.extend(item)
            else:
                self.state_images.append(item)
        return self

    def set_images(self, state_images: list["StateImage"]) -> "ObjectCollectionBuilder":
        """Set state images list.

        Args:
            state_images: List of StateImage objects

        Returns:
            This builder for method chaining
        """
        self.state_images = state_images
        return self

    def with_patterns(self, *patterns) -> "ObjectCollectionBuilder":
        """Add patterns as state images.

        Args:
            patterns: Variable number of Pattern objects or list

        Returns:
            This builder for method chaining
        """
        for item in patterns:
            if isinstance(item, list):
                for pattern in item:
                    self.state_images.append(pattern.in_null_state())
            else:
                self.state_images.append(item.in_null_state())
        return self

    def with_all_state_images(self, state: Optional["State"]) -> "ObjectCollectionBuilder":
        """Add all state images from a state.

        Args:
            state: State to get images from

        Returns:
            This builder for method chaining
        """
        if state is None:
            # ConsoleReporter.print("null state passed| ")
            return self
        else:
            self.state_images.extend(state.get_state_images())
        return self

    def with_non_shared_images(self, state: Optional["State"]) -> "ObjectCollectionBuilder":
        """Add non-shared state images from a state.

        Args:
            state: State to get images from

        Returns:
            This builder for method chaining
        """
        if state is None:
            # ConsoleReporter.print("null state passed| ")
            return self
        for state_image in state.get_state_images():
            if not state_image.is_shared():
                self.state_images.append(state_image)
        return self

    def with_regions(self, *regions) -> "ObjectCollectionBuilder":
        """Add regions to collection.

        Args:
            regions: Variable number of Region or StateRegion objects

        Returns:
            This builder for method chaining
        """
        for region in regions:
            if isinstance(region, Region):
                self.state_regions.append(region.in_null_state())
            elif isinstance(region, StateRegion):
                self.state_regions.append(region)
        return self

    def set_regions(self, regions: list["StateRegion"]) -> "ObjectCollectionBuilder":
        """Set regions list.

        Args:
            regions: List of StateRegion objects

        Returns:
            This builder for method chaining
        """
        self.state_regions = regions
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
        for region in regions:
            if isinstance(region, Region):
                for grid_region in region.get_grid_regions(rows, columns):
                    self.state_regions.append(grid_region.in_null_state())
            elif isinstance(region, StateRegion):
                for grid_region in region.get_search_region().get_grid_regions(rows, columns):
                    self.state_regions.append(grid_region.in_null_state())
        return self

    def with_strings(self, *strings) -> "ObjectCollectionBuilder":
        """Add strings to collection.

        Args:
            strings: Variable number of string or StateString objects

        Returns:
            This builder for method chaining
        """
        # Import locally to avoid circular dependency
        from ..model.state.state_string import StateString

        for string in strings:
            if isinstance(string, str):
                self.state_strings.append(StateString(string))
            elif isinstance(string, StateString):
                self.state_strings.append(string)
        return self

    def set_strings(self, strings: list["StateString"]) -> "ObjectCollectionBuilder":
        """Set strings list.

        Args:
            strings: List of StateString objects

        Returns:
            This builder for method chaining
        """
        self.state_strings = strings
        return self

    def with_matches(self, *matches: "ActionResult") -> "ObjectCollectionBuilder":
        """Add matches to collection.

        Args:
            matches: Variable number of ActionResult objects

        Returns:
            This builder for method chaining
        """
        self.matches.extend(matches)
        return self

    def set_matches(self, matches: list["ActionResult"]) -> "ObjectCollectionBuilder":
        """Set matches list.

        Args:
            matches: List of ActionResult objects

        Returns:
            This builder for method chaining
        """
        self.matches = matches
        return self

    def with_match_objects_as_regions(self, *matches: "Match") -> "ObjectCollectionBuilder":
        """Add match objects as regions.

        Args:
            matches: Variable number of Match objects

        Returns:
            This builder for method chaining
        """
        for match in matches:
            from ..model.state import StateRegionBuilder

            self.state_regions.append(
                StateRegionBuilder()
                .set_search_region(match.get_region())
                .set_owner_state_name("null")
                .build()
            )
        return self

    def with_match_objects_as_state_images(self, *matches: "Match") -> "ObjectCollectionBuilder":
        """Add match objects as state images.

        Args:
            matches: Variable number of Match objects

        Returns:
            This builder for method chaining
        """
        for match in matches:
            self.state_images.append(match.to_state_image())
        return self

    def with_scenes(self, *scenes) -> "ObjectCollectionBuilder":
        """Add scenes to collection.

        Args:
            scenes: Variable number of string, Pattern, Scene objects or list

        Returns:
            This builder for method chaining
        """
        for item in scenes:
            if isinstance(item, str):
                self.scenes.append(Scene(item))
            elif isinstance(item, Pattern):
                self.scenes.append(Scene(item))
            elif isinstance(item, Scene):
                self.scenes.append(item)
            elif isinstance(item, list):
                self.scenes.extend(item)
        return self

    def build(self) -> ObjectCollection:
        """Create the ObjectCollection with configured properties.

        Returns:
            Configured ObjectCollection instance
        """
        object_collection = ObjectCollection()
        object_collection.state_images = self.state_images.copy()
        object_collection.state_locations = self.state_locations.copy()
        object_collection.state_regions = self.state_regions.copy()
        object_collection.state_strings = self.state_strings.copy()
        object_collection.matches = self.matches.copy()
        object_collection.scenes = self.scenes.copy()
        return object_collection
