"""State object manager - manages state images, regions, locations, and strings.

Handles the collection and management of visual and interactive elements
that define and identify a state.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ...element.region import Region
from ...search_regions import SearchRegions

if TYPE_CHECKING:
    from ..state_image import StateImage
    from ..state_location import StateLocation
    from ..state_region import StateRegion
    from ..state_string import StateString


@dataclass
class StateObjectManager:
    """Manages state objects (images, regions, locations, strings).

    Responsible for storing and managing all visual and interactive elements
    that comprise a state's definition.
    """

    state_images: list[StateImage] = field(default_factory=list)
    """Visual patterns that identify this state."""

    state_regions: list[StateRegion] = field(default_factory=list)
    """Clickable/hoverable areas that can change state or retrieve text."""

    state_locations: list[StateLocation] = field(default_factory=list)
    """Specific points for precise interactions."""

    state_strings: list[StateString] = field(default_factory=list)
    """Text input fields that can change the expected state."""

    state_text: set[str] = field(default_factory=set)
    """Text that appears on screen as a clue to look for images in this state."""

    def add_image(self, state_image: StateImage, owner_state) -> None:
        """Add a StateImage to this state.

        Args:
            state_image: StateImage to add
            owner_state: The State object that owns this image
        """
        state_image.owner_state = owner_state
        self.state_images.append(state_image)

    def add_region(self, state_region: StateRegion, owner_state) -> None:
        """Add a StateRegion to this state.

        Args:
            state_region: StateRegion to add
            owner_state: The State object that owns this region
        """
        state_region.owner_state = owner_state
        self.state_regions.append(state_region)

    def add_location(self, state_location: StateLocation, owner_state) -> None:
        """Add a StateLocation to this state.

        Args:
            state_location: StateLocation to add
            owner_state: The State object that owns this location
        """
        state_location.owner_state = owner_state
        self.state_locations.append(state_location)

    def add_string(self, state_string: StateString, owner_state) -> None:
        """Add a StateString to this state.

        Args:
            state_string: StateString to add
            owner_state: The State object that owns this string
        """
        state_string.owner_state = owner_state
        self.state_strings.append(state_string)

    def add_text(self, text: str) -> None:
        """Add state text.

        Args:
            text: Text to add
        """
        self.state_text.add(text)

    def get_images(self) -> list[StateImage]:
        """Get list of state images.

        Returns:
            List of StateImage objects
        """
        return self.state_images

    def get_regions(self) -> list[StateRegion]:
        """Get list of state regions.

        Returns:
            List of StateRegion objects
        """
        return self.state_regions

    def get_locations(self) -> list[StateLocation]:
        """Get list of state locations.

        Returns:
            List of StateLocation objects
        """
        return self.state_locations

    def get_strings(self) -> list[StateString]:
        """Get list of state strings.

        Returns:
            List of StateString objects
        """
        return self.state_strings

    def set_search_region_for_all_images(self, search_region: Region) -> None:
        """Set the search region for all images.

        Args:
            search_region: Region to set
        """
        search_regions = SearchRegions().add_region(search_region)
        for image_obj in self.state_images:
            image_obj.set_search_regions(search_regions)

    def get_boundaries(self) -> Region:
        """Get the boundaries of the state using StateRegion, StateImage, and StateLocation objects.

        Snapshots and SearchRegion(s) are used for StateImages.

        Returns:
            The boundaries of the state
        """
        image_regions = []

        # Add regions from StateImages
        for state_image in self.state_images:
            # Add fixed regions from search_regions
            if state_image.search_regions:
                fixed_region = state_image.search_regions.get_fixed_region()
                if fixed_region and fixed_region.is_defined():
                    image_regions.append(fixed_region)

            # Add snapshot locations
            snapshots = state_image.get_all_match_snapshots()
            for snapshot in snapshots:
                for match in snapshot.match_list:
                    image_regions.append(match.get_region())

        # Add regions from StateRegions
        for state_region in self.state_regions:
            image_regions.append(state_region.search_region)

        # Add regions from StateLocations
        for state_location in self.state_locations:
            loc = state_location.location
            final_loc = loc.get_final_location()
            image_regions.append(Region(x=final_loc.x, y=final_loc.y, width=0, height=0))

        if not image_regions:
            return Region()  # Return undefined region

        # Calculate union of all regions
        union = image_regions[0]
        for i in range(1, len(image_regions)):
            union = union.union(image_regions[i])

        return union

    def __str__(self) -> str:
        """String representation."""
        parts = []
        parts.append(f"Images={len(self.state_images)}")
        for img in self.state_images:
            parts.append(f"  {img}")
        parts.append(f"Regions={len(self.state_regions)}")
        for reg in self.state_regions:
            parts.append(f"  {reg}")
        parts.append(f"Locations={len(self.state_locations)}")
        for loc in self.state_locations:
            parts.append(f"  {loc}")
        parts.append(f"Strings={len(self.state_strings)}")
        for s in self.state_strings:
            parts.append(f"  {s}")
        return "\n".join(parts)
