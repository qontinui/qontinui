"""Object collection - ported from Qontinui framework.

Container for GUI elements serving as action targets.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model.element import Scene
    from ..model.state import StateImage, StateLocation, StateRegion, StateString
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

        Note: ActionResult instances in matches are immutable and cannot be modified.
        Their times_acted_on value is set during construction via ActionResultBuilder.
        """
        for sio in self.state_images:
            sio.set_times_acted_on(0)
        for sl in self.state_locations:
            sl.set_times_acted_on(0)
        for sr in self.state_regions:
            sr.set_times_acted_on(0)
        for ss in self.state_strings:
            ss.set_times_acted_on(0)
        # Skip matches as ActionResult is immutable

    def get_first_object_name(self) -> str:
        """Get name of first object in collection.

        Returns:
            Name of first object or empty string
        """
        if self.state_images:
            name = self.state_images[0].get_name()
            if name:
                return str(name)
            patterns = self.state_images[0].get_patterns()
            if patterns and patterns[0].get_imgpath():
                imgpath = patterns[0].get_imgpath()
                return str(imgpath)
        if self.state_locations:
            loc_name = self.state_locations[0].get_name()
            if loc_name:
                return str(loc_name)
        if self.state_regions:
            reg_name = self.state_regions[0].get_name()
            if reg_name:
                return str(reg_name)
        if self.state_strings:
            string_val = self.state_strings[0].get_string()
            return str(string_val)
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
            Set of unique image filenames (empty strings filtered out)
        """
        filenames = set()
        for si in self.state_images:
            for p in si.get_patterns():
                imgpath = p.get_imgpath()
                if imgpath:  # Only add non-None, non-empty paths
                    filenames.add(imgpath)
        return filenames

    def get_all_owner_states(self) -> set[str]:
        """Get all owner state names.

        Returns:
            Set of unique owner state names
        """
        states = set()
        for si in self.state_images:
            states.add(si.get_owner_state_name())
        for sl in self.state_locations:
            states.add(sl.get_owner_state_name())
        for sr in self.state_regions:
            states.add(sr.get_owner_state_name())
        for ss in self.state_strings:
            states.add(ss.get_owner_state_name())
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


# Import ObjectCollectionBuilder from builders package
from .builders import ObjectCollectionBuilder

# Add Builder class attribute to support Brobot's nested class pattern
# This allows: ObjectCollection.Builder() instead of ObjectCollectionBuilder()
ObjectCollection.Builder = ObjectCollectionBuilder  # type: ignore[attr-defined]
