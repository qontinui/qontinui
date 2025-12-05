"""StateRegion - ported from Qontinui framework.

Regions associated with states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..element import Location, Region
from .action_history import ActionHistory

if TYPE_CHECKING:
    from qontinui.actions.action_result import ActionResult
    from qontinui.model.state.state import State


@dataclass
class StateRegion:
    """Region associated with a state.

    Port of StateRegion from Qontinui framework class.
    Represents a region that is part of a state.
    """

    region: Region
    name: str | None = None
    owner_state: State | None = None

    # Region properties
    _fixed: bool = True  # If true, region is fixed in position
    _search_region: bool = False  # If true, used as search region for state
    _interaction_region: bool = False  # If true, used for interactions

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Action history for integration testing
    action_history: ActionHistory = field(default_factory=ActionHistory)

    def __post_init__(self):
        """Initialize region name."""
        if self.name is None:
            self.name = f"Region_{self.region.x}_{self.region.y}"

    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is in region.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is in region
        """
        from ..element import Location

        return self.region.contains(Location(x, y))

    def get_center(self) -> Location:
        """Get center of region.

        Returns:
            Center location
        """
        return self.region.center

    def click(self) -> ActionResult:
        """Click in the center of this region.

        Returns:
            ActionResult from click
        """
        from ..actions import Action
        from ..actions.basic.click.click_options import ClickOptionsBuilder

        center = self.get_center()
        click_options = ClickOptionsBuilder().build()
        action = Action(click_options)
        result: ActionResult = action.click(center.x, center.y)
        return result

    def hover(self) -> ActionResult:
        """Move mouse to center of region.

        Returns:
            ActionResult from move
        """
        from ..actions import Action

        center = self.get_center()
        action = Action()
        result: ActionResult = action.move(center.x, center.y)
        return result

    def set_fixed(self, fixed: bool = True) -> StateRegion:
        """Set whether region is fixed (fluent).

        Args:
            fixed: True if region is fixed

        Returns:
            Self for chaining
        """
        self._fixed = fixed
        return self

    def set_search_region(self, search: bool = True) -> StateRegion:
        """Set whether this is a search region (fluent).

        Args:
            search: True if search region

        Returns:
            Self for chaining
        """
        self._search_region = search
        return self

    def set_interaction_region(self, interaction: bool = True) -> StateRegion:
        """Set whether this is an interaction region (fluent).

        Args:
            interaction: True if interaction region

        Returns:
            Self for chaining
        """
        self._interaction_region = interaction
        return self

    @property
    def is_fixed(self) -> bool:
        """Check if region is fixed."""
        return self._fixed

    @property
    def is_search_region(self) -> bool:
        """Check if this is a search region."""
        return self._search_region

    @property
    def is_interaction_region(self) -> bool:
        """Check if this is an interaction region."""
        return self._interaction_region

    @property
    def search_region(self) -> Region:
        """Get the region itself when used as a search region.

        Returns:
            The region
        """
        return self.region

    def get_name(self) -> str:
        """Get the name of this state region.

        Returns:
            Name or empty string
        """
        return self.name or ""

    def get_owner_state_name(self) -> str:
        """Get the owner state name.

        Returns:
            Owner state name or empty string
        """
        return self.owner_state.name if self.owner_state else ""

    def get_search_region(self) -> Region:
        """Get the search region (the region itself).

        Returns:
            The region
        """
        return self.region

    def set_times_acted_on(self, times: int) -> None:
        """Set times this region has been acted upon.

        Args:
            times: Number of times acted upon
        """
        # This would typically update action history
        # For now, just a placeholder
        pass

    def __str__(self) -> str:
        """String representation."""
        state_name = self.owner_state.name if self.owner_state else "None"
        return f"StateRegion('{self.name}' in state '{state_name}')"
