"""State metrics manager - tracks visits, probability, and statistics.

Manages metrics related to state usage, including visit counts,
existence probability, and stochastic testing modifiers.
"""

from dataclasses import dataclass, field
from datetime import datetime

from ....model.element.scene import Scene
from ...element.region import Region
from ..action_history import ActionHistory


@dataclass
class StateMetricsManager:
    """Manages state metrics and statistics.

    Responsible for tracking visit counts, probability calculations,
    stochastic testing parameters, and action history.
    """

    times_visited: int = 0
    """Number of times this state has been visited."""

    probability_exists: int = 0
    """Current probability that the state exists (used for mocks)."""

    base_mock_find_stochastic_modifier: int = 100
    """Base probability modifier for mock find operations (stochastic testing)."""

    path_score: int = 1
    """Larger path scores discourage taking a path with this state."""

    last_accessed: datetime | None = None
    """When this state was last accessed."""

    is_initial: bool = False
    """Whether this is an initial/starting state."""

    scenes: list[Scene] = field(default_factory=list)
    """Screenshots where the state is found."""

    usable_area: Region = field(default_factory=Region)
    """The region used to find images."""

    match_history: ActionHistory = field(default_factory=ActionHistory)
    """History of actions performed in this state."""

    def add_visit(self) -> None:
        """Increment visit counter and update last accessed time."""
        self.times_visited += 1
        self.last_accessed = datetime.now()

    def get_visit_count(self) -> int:
        """Get the number of times this state has been visited.

        Returns:
            Number of visits
        """
        return self.times_visited

    def reset_visits(self) -> None:
        """Reset visit counter."""
        self.times_visited = 0

    def set_probability_exists(self, probability: int) -> None:
        """Set the existence probability.

        Args:
            probability: Probability value (0-100)
        """
        self.probability_exists = probability

    def get_probability_exists(self) -> int:
        """Get the existence probability.

        Returns:
            Probability value (0-100)
        """
        return self.probability_exists

    def set_probability_to_base_probability(self) -> None:
        """Reset probability to base probability."""
        self.probability_exists = self.base_mock_find_stochastic_modifier

    def set_base_mock_find_stochastic_modifier(self, probability: int) -> None:
        """Set base mock find stochastic modifier.

        Args:
            probability: Base probability modifier (0-100)
        """
        self.base_mock_find_stochastic_modifier = probability

    def get_base_mock_find_stochastic_modifier(self) -> int:
        """Get base mock find stochastic modifier.

        Returns:
            Base probability modifier (0-100)
        """
        return self.base_mock_find_stochastic_modifier

    def set_path_score(self, score: int) -> None:
        """Set the path score.

        Args:
            score: Path score value (higher = less preferred)
        """
        self.path_score = score

    def get_path_score(self) -> int:
        """Get the path score.

        Returns:
            Path score value
        """
        return self.path_score

    def set_is_initial(self, is_initial: bool) -> None:
        """Set whether this is an initial state.

        Args:
            is_initial: True if this is an initial/starting state
        """
        self.is_initial = is_initial

    def is_initial_state(self) -> bool:
        """Check if this is an initial state.

        Returns:
            True if this is an initial state
        """
        return self.is_initial

    def get_last_accessed(self) -> datetime | None:
        """Get the last access time.

        Returns:
            Last accessed datetime or None if never accessed
        """
        return self.last_accessed

    def add_scene(self, scene: Scene) -> None:
        """Add a scene (screenshot) where this state was found.

        Args:
            scene: Scene to add
        """
        self.scenes.append(scene)

    def get_scenes(self) -> list[Scene]:
        """Get all scenes where this state was found.

        Returns:
            List of scenes
        """
        return self.scenes

    def set_usable_area(self, area: Region) -> None:
        """Set the usable area for finding images.

        Args:
            area: Usable area region
        """
        self.usable_area = area

    def get_usable_area(self) -> Region:
        """Get the usable area.

        Returns:
            Usable area region
        """
        return self.usable_area

    def get_match_history(self) -> ActionHistory:
        """Get the action history for this state.

        Returns:
            ActionHistory instance
        """
        return self.match_history

    def __str__(self) -> str:
        """String representation."""
        parts = []
        parts.append(f"Visits: {self.times_visited}")
        parts.append(f"Probability exists: {self.probability_exists}%")
        parts.append(f"Path score: {self.path_score}")
        parts.append(f"Is initial: {self.is_initial}")
        if self.last_accessed:
            parts.append(f"Last accessed: {self.last_accessed.isoformat()}")
        parts.append(f"Scenes: {len(self.scenes)}")
        parts.append(f"Action history: {self.match_history}")
        return "\n".join(parts)
