"""State model - ported from Qontinui framework.

Represents a distinct configuration of the GUI in the model-based automation framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, cast

from ..element.region import Region
from ..search_regions import SearchRegions

if TYPE_CHECKING:
    from qontinui.model.element.scene import Scene
    from qontinui.model.state.state_enum import StateEnum
    from qontinui.model.state.state_image import StateImage
    from qontinui.model.state.state_location import StateLocation
    from qontinui.model.state.state_region import StateRegion
    from qontinui.model.state.state_string import StateString
    from qontinui.model.transition.state_transition import StateTransition


@dataclass
class State:
    """Represents a distinct configuration of the GUI.

    Port of State from Qontinui framework class.

    A State is the fundamental building block of the model-based approach, representing a
    recognizable and meaningful configuration of the user interface. States form the nodes in
    the state structure (Ω), which models the GUI environment as a navigable graph.

    Key concepts:
    - Identification: States are identified by their StateImages - visual patterns
      that uniquely define this GUI configuration
    - Navigation: States are connected by transitions, allowing Brobot to navigate
      between them like pages in a website
    - Recovery: If Brobot gets lost, it can identify the current State and find
      a new path to the target State
    - Hierarchy: States can hide other States (like menus covering content) and
      can be blocking (requiring interaction before accessing other States)

    State components:
    - StateImages: Visual patterns that identify this State (some may be shared across States)
    - StateRegions: Clickable or hoverable areas that can trigger State changes
    - StateStrings: Text input fields that affect State transitions
    - StateLocations: Specific points for precise interactions
    - StateText: Text that appears in this State (used for faster preliminary searches)

    This class embodies the core principle of model-based GUI automation: transforming
    implicit knowledge about GUI structure into an explicit, navigable model that enables
    robust and maintainable automation.
    """

    name: str = ""
    """Name of this state."""

    description: str = ""
    """Optional description of this state."""

    id: int | None = None
    """Database ID, set when saved."""

    state_enum: StateEnum | None = None
    """Optional enum value for this state."""

    state_text: set[str] = field(default_factory=set)
    """Text that appears on screen as a clue to look for images in this state."""

    state_images: list[StateImage] = field(default_factory=list)
    """Visual patterns that identify this state."""

    state_strings: list[StateString] = field(default_factory=list)
    """Text input fields that can change the expected state."""

    state_regions: list[StateRegion] = field(default_factory=list)
    """Clickable/hoverable areas that can change state or retrieve text."""

    state_locations: list[StateLocation] = field(default_factory=list)
    """Specific points for precise interactions."""

    blocking: bool = False
    """When true, this State needs to be acted on before accessing other States."""

    can_hide: set[str] = field(default_factory=set)
    """States that this State can hide when it becomes active."""

    can_hide_ids: set[int] = field(default_factory=set)
    """IDs of states that this State can hide."""

    hidden_state_names: set[str] = field(default_factory=set)
    """Names of currently hidden states (used when initializing in code)."""

    hidden_state_ids: set[int] = field(default_factory=set)
    """IDs of currently hidden states (used at runtime)."""

    path_score: int = 1
    """Larger path scores discourage taking a path with this state."""

    last_accessed: datetime | None = None
    """When this state was last accessed."""

    is_initial: bool = False
    """Whether this is an initial/starting state."""


    baseMockFindStochasticModifier: int = 100
    """Base probability modifier for mock find operations (stochastic testing)."""

    probability_exists: int = 0
    """Current probability that the state exists (used for mocks)."""

    times_visited: int = 0
    """Number of times this state has been visited."""

    scenes: list[Scene] = field(default_factory=list)
    """Screenshots where the state is found."""

    usable_area: Region = field(default_factory=Region)
    """The region used to find images."""

    match_history: ActionHistory = field(default_factory=lambda: ActionHistory())
    """History of actions performed in this state."""

    transitions: list[StateTransition] = field(default_factory=list)
    """List of transitions from this state."""

    def exists(self, timeout: float = 0.0) -> bool:
        """Check if this state exists.

        Args:
            timeout: Maximum time to wait for state to exist

        Returns:
            True if state exists, False otherwise
        """
        # This is a placeholder implementation
        # The actual implementation should check if state images are visible
        return self.probability_exists > 0

    def wait_for(self, timeout: float = 10.0) -> bool:
        """Wait for this state to appear.

        Args:
            timeout: Maximum time to wait

        Returns:
            True if state appeared, False otherwise
        """
        # This is a placeholder implementation
        # The actual implementation should wait for state images to be visible
        return self.exists(timeout)

    def get_transitions_to(self, target_state: str) -> list[StateTransition]:
        """Get transitions to a specific target state.

        Args:
            target_state: Target state name

        Returns:
            List of transitions to the target state
        """
        return [t for t in self.transitions if t.to_state == target_state]

    def get_possible_next_states(self) -> list[str]:
        """Get list of possible next states from this state.

        Returns:
            List of state names that can be reached from this state
        """
        next_states = []
        for transition in self.transitions:
            if transition.to_state:
                next_states.append(transition.to_state)
        return list(set(next_states))  # Remove duplicates

    def add_transition(self, transition: StateTransition) -> None:
        """Add a transition from this state.

        Args:
            transition: StateTransition to add
        """
        self.transitions.append(transition)

    def get_state_images(self) -> list[StateImage]:
        """Get list of state images.

        Returns:
            List of StateImage objects
        """
        return self.state_images

    def add_state_image(self, state_image: StateImage) -> None:
        """Add a StateImage to this state.

        Args:
            state_image: StateImage to add
        """
        state_image.owner_state = self
        self.state_images.append(state_image)

    def add_state_region(self, state_region: StateRegion) -> None:
        """Add a StateRegion to this state.

        Args:
            state_region: StateRegion to add
        """
        state_region.owner_state = self
        self.state_regions.append(state_region)

    def add_state_location(self, state_location: StateLocation) -> None:
        """Add a StateLocation to this state.

        Args:
            state_location: StateLocation to add
        """
        state_location.owner_state = self
        self.state_locations.append(state_location)

    def add_state_string(self, state_string: StateString) -> None:
        """Add a StateString to this state.

        Args:
            state_string: StateString to add
        """
        state_string.owner_state = self
        self.state_strings.append(state_string)

    def add_state_text(self, text: str) -> None:
        """Add state text.

        Args:
            text: Text to add
        """
        self.state_text.add(text)

    def set_search_region_for_all_images(self, search_region: Region) -> None:
        """Set the search region for all images.

        Args:
            search_region: Region to set
        """
        search_regions = SearchRegions().add_region(search_region)
        for image_obj in self.state_images:
            image_obj.set_search_regions(search_regions)

    def set_probability_to_base_probability(self) -> None:
        """Reset probability to base probability."""
        self.probability_exists = self.baseMockFindStochasticModifier

    def add_hidden_state(self, state_id: int) -> None:
        """Add a hidden state ID.

        Args:
            state_id: ID of state to hide
        """
        self.hidden_state_ids.add(state_id)

    def reset_hidden(self) -> None:
        """Reset hidden state names."""
        self.hidden_state_names = set()

    def add_visit(self) -> None:
        """Increment visit counter."""
        self.times_visited += 1

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

        return cast(Region, union)

    def __str__(self) -> str:
        """String representation."""
        parts = [f"State: {self.name}"]
        parts.append(f"Images={len(self.state_images)}")
        for img in self.state_images:
            parts.append(str(img))
        parts.append(f"Regions={len(self.state_regions)}")
        for reg in self.state_regions:
            parts.append(str(reg))
        parts.append(f"Locations={len(self.state_locations)}")
        for loc in self.state_locations:
            parts.append(str(loc))
        parts.append(f"Strings={len(self.state_strings)}")
        for s in self.state_strings:
            parts.append(str(s))
        return "\n".join(parts)


class StateBuilder:
    """Builder for creating State objects.

    Port of State from Qontinui framework.Builder class.
    """

    def __init__(self, name: str):
        """Initialize builder with state name.

        Args:
            name: Name of the state
        """
        self.name = name
        self.description = ""
        self.state_text: set[str] = set()
        self.state_images: list[StateImage] = []  # List instead of set for unhashable objects
        self.state_strings: list[StateString] = []  # List instead of set for unhashable objects
        self.state_regions: list[StateRegion] = []  # List instead of set for unhashable objects
        self.state_locations: list[StateLocation] = []  # List instead of set for unhashable objects
        self.blocking = False
        self.can_hide: set[str] = set()
        self.hidden: set[str] = set()
        self.path_score = 1
        self.last_accessed = None
        self.is_initial = False
        self.baseMockFindStochasticModifier = 100
        self.scenes: list[Scene] = []
        self.usable_area = Region()

    def with_description(self, description: str) -> StateBuilder:
        """Set state description.

        Args:
            description: Description of the state

        Returns:
            Self for chaining
        """
        self.description = description
        return self

    def with_text(self, *state_text: str) -> StateBuilder:
        """Add state text.

        Args:
            *state_text: Text strings to add

        Returns:
            Self for chaining
        """
        self.state_text.update(state_text)
        return self

    def with_images(self, *state_images: StateImage) -> StateBuilder:
        """Add state images.

        Args:
            *state_images: StateImages to add

        Returns:
            Self for chaining
        """
        self.state_images.extend(state_images)
        return self

    def with_strings(self, *state_strings: StateString) -> StateBuilder:
        """Add state strings.

        Args:
            *state_strings: StateStrings to add

        Returns:
            Self for chaining
        """
        self.state_strings.extend(state_strings)
        return self

    def with_regions(self, *state_regions: StateRegion) -> StateBuilder:
        """Add state regions.

        Args:
            *state_regions: StateRegions to add

        Returns:
            Self for chaining
        """
        self.state_regions.extend(state_regions)
        return self

    def with_locations(self, *state_locations: StateLocation) -> StateBuilder:
        """Add state locations.

        Args:
            *state_locations: StateLocations to add

        Returns:
            Self for chaining
        """
        self.state_locations.extend(state_locations)
        return self

    def set_blocking(self, blocking: bool) -> StateBuilder:
        """Set blocking flag.

        Args:
            blocking: Whether state is blocking

        Returns:
            Self for chaining
        """
        self.blocking = blocking
        return self

    def can_hide_states(self, *state_names: str) -> StateBuilder:
        """Add states that can be hidden.

        Args:
            *state_names: Names of states that can be hidden

        Returns:
            Self for chaining
        """
        self.can_hide.update(state_names)
        return self

    def set_path_score(self, score: int) -> StateBuilder:
        """Set path score.

        Args:
            score: Path score value

        Returns:
            Self for chaining
        """
        self.path_score = score
        return self

    def set_base_mock_find_stochastic_modifier(self, probability: int) -> StateBuilder:
        """Set base mock find stochastic modifier.

        Args:
            probability: Base probability modifier for mock find operations (0-100)

        Returns:
            Self for chaining
        """
        self.baseMockFindStochasticModifier = probability
        return self

    def set_is_initial(self, is_initial: bool) -> StateBuilder:
        """Set whether this is an initial state.

        Args:
            is_initial: True if this is an initial/starting state

        Returns:
            Self for chaining
        """
        self.is_initial = is_initial
        return self


    def with_scenes(self, *scenes: Scene) -> StateBuilder:
        """Add scenes.

        Args:
            *scenes: Scenes to add

        Returns:
            Self for chaining
        """
        self.scenes.extend(scenes)
        return self

    def set_usable_area(self, area: Region) -> StateBuilder:
        """Set usable area.

        Args:
            area: Usable area region

        Returns:
            Self for chaining
        """
        self.usable_area = area
        return self

    def build(self) -> State:
        """Build the State object.

        Returns:
            Constructed State
        """
        state = State(name=self.name, description=self.description)

        # Set all fields
        state.state_text = self.state_text
        state.blocking = self.blocking
        state.can_hide = self.can_hide
        state.hidden_state_names = self.hidden
        state.path_score = self.path_score
        state.last_accessed = self.last_accessed
        state.is_initial = self.is_initial
        state.baseMockFindStochasticModifier = self.baseMockFindStochasticModifier
        state.probability_exists = 0
        state.scenes = self.scenes
        state.usable_area = self.usable_area

        # Add state objects (which sets owner state name)
        for img in self.state_images:
            state.add_state_image(img)
        for reg in self.state_regions:
            state.add_state_region(reg)
        for loc in self.state_locations:
            state.add_state_location(loc)
        for s in self.state_strings:
            state.add_state_string(s)

        return state


class ActionHistory:
    """Placeholder for ActionHistory class.

    Will be implemented when migrating the action package.
    """

    pass
