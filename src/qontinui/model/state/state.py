"""State model - ported from Qontinui framework.

Represents a distinct configuration of the GUI in the model-based automation framework.
"""


from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from ..element.region import Region
from .managers import (
    StateMetricsManager,
    StateObjectManager,
    StateTransitionManager,
    StateVisibilityManager,
)

if TYPE_CHECKING:
    from qontinui.model.element.scene import Scene
    from qontinui.model.state.state_enum import StateEnum
    from qontinui.model.state.state_image import StateImage
    from qontinui.model.state.state_location import StateLocation
    from qontinui.model.state.state_region import StateRegion
    from qontinui.model.state.state_string import StateString
    from qontinui.model.transition.state_transition import StateTransition

    from .action_history import ActionHistory


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

    Architecture:
    State is an aggregate root following DDD principles, delegating to specialized
    managers for different concerns:
    - StateObjectManager: Manages state objects (images, regions, locations, strings)
    - StateTransitionManager: Manages transitions between states
    - StateVisibilityManager: Handles hiding/blocking logic
    - StateMetricsManager: Tracks visits, probability, statistics
    """

    name: str = ""
    """Name of this state."""

    description: str = ""
    """Optional description of this state."""

    id: int | None = None
    """Database ID, set when saved."""

    state_enum: StateEnum | None = None
    """Optional enum value for this state."""

    # Composed managers (DDD aggregate root pattern)
    _objects: StateObjectManager = field(default_factory=StateObjectManager)
    """Manages state objects (images, regions, locations, strings)."""

    _transitions: StateTransitionManager = field(default_factory=StateTransitionManager)
    """Manages transitions from and to this state."""

    _visibility: StateVisibilityManager = field(default_factory=StateVisibilityManager)
    """Manages state visibility, hiding, and blocking."""

    _metrics: StateMetricsManager = field(default_factory=StateMetricsManager)
    """Manages state metrics and statistics."""

    # Properties that delegate to managers
    @property
    def state_text(self) -> set[str]:
        """Text that appears on screen as a clue to look for images in this state."""
        return self._objects.state_text

    @property
    def state_images(self) -> list[StateImage]:
        """Visual patterns that identify this state."""
        return self._objects.state_images

    @property
    def state_strings(self) -> list[StateString]:
        """Text input fields that can change the expected state."""
        return self._objects.state_strings

    @property
    def state_regions(self) -> list[StateRegion]:
        """Clickable/hoverable areas that can change state or retrieve text."""
        return self._objects.state_regions

    @property
    def state_locations(self) -> list[StateLocation]:
        """Specific points for precise interactions."""
        return self._objects.state_locations

    @property
    def blocking(self) -> bool:
        """When true, this State needs to be acted on before accessing other States."""
        return self._visibility.blocking

    @blocking.setter
    def blocking(self, value: bool) -> None:
        """Set blocking state."""
        self._visibility.blocking = value

    @property
    def can_hide(self) -> set[str]:
        """States that this State can hide when it becomes active."""
        return self._visibility.can_hide

    @property
    def can_hide_ids(self) -> set[int]:
        """IDs of states that this State can hide."""
        return self._visibility.can_hide_ids

    @property
    def hidden_state_names(self) -> set[str]:
        """Names of currently hidden states (used when initializing in code)."""
        return self._visibility.hidden_state_names

    @hidden_state_names.setter
    def hidden_state_names(self, value: set[str]) -> None:
        """Set hidden state names."""
        self._visibility.hidden_state_names = value

    @property
    def hidden_state_ids(self) -> set[int]:
        """IDs of currently hidden states (used at runtime)."""
        return self._visibility.hidden_state_ids

    @property
    def path_score(self) -> int:
        """Larger path scores discourage taking a path with this state."""
        return self._metrics.path_score

    @path_score.setter
    def path_score(self, value: int) -> None:
        """Set path score."""
        self._metrics.path_score = value

    @property
    def last_accessed(self) -> datetime | None:
        """When this state was last accessed."""
        return self._metrics.last_accessed

    @last_accessed.setter
    def last_accessed(self, value: datetime | None) -> None:
        """Set last accessed time."""
        self._metrics.last_accessed = value

    @property
    def is_initial(self) -> bool:
        """Whether this is an initial/starting state."""
        return self._metrics.is_initial

    @is_initial.setter
    def is_initial(self, value: bool) -> None:
        """Set is_initial flag."""
        self._metrics.is_initial = value

    @property
    def baseMockFindStochasticModifier(self) -> int:
        """Base probability modifier for mock find operations (stochastic testing)."""
        return self._metrics.base_mock_find_stochastic_modifier

    @baseMockFindStochasticModifier.setter
    def baseMockFindStochasticModifier(self, value: int) -> None:
        """Set base mock find stochastic modifier."""
        self._metrics.base_mock_find_stochastic_modifier = value

    @property
    def probability_exists(self) -> int:
        """Current probability that the state exists (used for mocks)."""
        return self._metrics.probability_exists

    @probability_exists.setter
    def probability_exists(self, value: int) -> None:
        """Set probability exists."""
        self._metrics.probability_exists = value

    @property
    def times_visited(self) -> int:
        """Number of times this state has been visited."""
        return self._metrics.times_visited

    @times_visited.setter
    def times_visited(self, value: int) -> None:
        """Set times visited."""
        self._metrics.times_visited = value

    @property
    def scenes(self) -> list[Scene]:
        """Screenshots where the state is found."""
        return self._metrics.scenes

    @scenes.setter
    def scenes(self, value: list[Scene]) -> None:
        """Set scenes."""
        self._metrics.scenes = value

    @property
    def usable_area(self) -> Region:
        """The region used to find images."""
        return self._metrics.usable_area

    @usable_area.setter
    def usable_area(self, value: Region) -> None:
        """Set usable area."""
        self._metrics.usable_area = value

    @property
    def match_history(self) -> ActionHistory:
        """History of actions performed in this state."""
        return self._metrics.match_history

    @property
    def transitions(self) -> list[StateTransition]:
        """List of outgoing transitions from this state."""
        return self._transitions.transitions

    @property
    def incoming_transitions(self) -> list[StateTransition]:
        """List of incoming transitions to this state (verification workflows executed when entering)."""
        return self._transitions.incoming_transitions

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

    # Delegation methods - Transitions
    def get_transitions_to(self, target_state: str) -> list[StateTransition]:
        """Get transitions to a specific target state.

        Args:
            target_state: Target state name

        Returns:
            List of transitions to the target state
        """
        return self._transitions.get_transitions_to(target_state)

    def get_possible_next_states(self) -> list[str]:
        """Get list of possible next states from this state.

        Returns:
            List of state names that can be reached from this state
        """
        return self._transitions.get_possible_next_states()

    def add_transition(self, transition: StateTransition) -> None:
        """Add a transition from this state.

        Args:
            transition: StateTransition to add
        """
        self._transitions.add(transition)

    # Delegation methods - Objects
    def get_state_images(self) -> list[StateImage]:
        """Get list of state images.

        Returns:
            List of StateImage objects
        """
        return self._objects.get_images()

    def add_state_image(self, state_image: StateImage) -> None:
        """Add a StateImage to this state.

        Args:
            state_image: StateImage to add
        """
        self._objects.add_image(state_image, self)

    def add_state_region(self, state_region: StateRegion) -> None:
        """Add a StateRegion to this state.

        Args:
            state_region: StateRegion to add
        """
        self._objects.add_region(state_region, self)

    def add_state_location(self, state_location: StateLocation) -> None:
        """Add a StateLocation to this state.

        Args:
            state_location: StateLocation to add
        """
        self._objects.add_location(state_location, self)

    def add_state_string(self, state_string: StateString) -> None:
        """Add a StateString to this state.

        Args:
            state_string: StateString to add
        """
        self._objects.add_string(state_string, self)

    def add_state_text(self, text: str) -> None:
        """Add state text.

        Args:
            text: Text to add
        """
        self._objects.add_text(text)

    def set_search_region_for_all_images(self, search_region: Region) -> None:
        """Set the search region for all images.

        Args:
            search_region: Region to set
        """
        self._objects.set_search_region_for_all_images(search_region)

    def get_boundaries(self) -> Region:
        """Get the boundaries of the state using StateRegion, StateImage, and StateLocation objects.

        Snapshots and SearchRegion(s) are used for StateImages.

        Returns:
            The boundaries of the state
        """
        return self._objects.get_boundaries()

    # Delegation methods - Visibility
    def add_hidden_state(self, state_id: int) -> None:
        """Add a hidden state ID.

        Args:
            state_id: ID of state to hide
        """
        self._visibility.add_hidden_state(state_id)

    def reset_hidden(self) -> None:
        """Reset hidden state names."""
        self._visibility.reset_hidden()

    # Delegation methods - Metrics
    def set_probability_to_base_probability(self) -> None:
        """Reset probability to base probability."""
        self._metrics.set_probability_to_base_probability()

    def add_visit(self) -> None:
        """Increment visit counter."""
        self._metrics.add_visit()

    def __str__(self) -> str:
        """String representation."""
        parts = [f"State: {self.name}"]
        parts.append(str(self._objects))
        return "\n".join(parts)


class StateBuilder:
    """Builder for creating State objects.

    Port of State from Qontinui framework.Builder class.
    """

    def __init__(self, name: str) -> None:
        """Initialize builder with state name.

        Args:
            name: Name of the state
        """
        self.name = name
        self.description = ""
        self.state_text: set[str] = set()
        self.state_images: list[StateImage] = []
        self.state_strings: list[StateString] = []
        self.state_regions: list[StateRegion] = []
        self.state_locations: list[StateLocation] = []
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

        # Set visibility properties
        state.blocking = self.blocking
        state._visibility.can_hide = self.can_hide
        state.hidden_state_names = self.hidden

        # Set metrics properties
        state.path_score = self.path_score
        state.last_accessed = self.last_accessed
        state.is_initial = self.is_initial
        state.baseMockFindStochasticModifier = self.baseMockFindStochasticModifier
        state.probability_exists = 0
        state.scenes = self.scenes
        state.usable_area = self.usable_area

        # Add state text
        for text in self.state_text:
            state.add_state_text(text)

        # Add state objects (which sets owner state)
        for img in self.state_images:
            state.add_state_image(img)
        for reg in self.state_regions:
            state.add_state_region(reg)
        for loc in self.state_locations:
            state.add_state_location(loc)
        for s in self.state_strings:
            state.add_state_string(s)

        return state
