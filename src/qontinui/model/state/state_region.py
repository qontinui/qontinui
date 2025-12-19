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

    # Monitor association - which monitor(s) this region is on
    # Required for multi-monitor support. Must be a specific monitor (not -1/all).
    monitors: list[int] = field(default_factory=lambda: [0])

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Action history for integration testing
    action_history: ActionHistory = field(default_factory=ActionHistory)

    def __post_init__(self):
        """Initialize region name."""
        if self.name is None:
            self.name = f"Region_{self.region.x}_{self.region.y}"

    def _get_global_coordinates(self, x: int, y: int) -> tuple[int, int]:
        """Convert coordinates to global screen coordinates.

        If the region has monitor associations, the coordinates are assumed
        to be relative to that monitor and are translated to global coordinates.

        Args:
            x: X coordinate (monitor-relative)
            y: Y coordinate (monitor-relative)

        Returns:
            (x, y) tuple in global screen coordinates
        """
        if self.monitors and self.monitors[0] >= 0:
            try:
                from qontinui.monitor.monitor_manager import MonitorManager

                manager = MonitorManager()
                monitor_idx = self.monitors[0]

                # Always translate coordinates - monitor 0 may not be at (0,0)
                # This handles cases where monitor 0 is positioned to the right or below other monitors
                return manager.to_global_coordinates(x, y, monitor_idx)
            except Exception:
                # Fall back to using coordinates as-is
                pass

        return (x, y)

    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is in region.

        Args:
            x: X coordinate (in global screen coordinates)
            y: Y coordinate (in global screen coordinates)

        Returns:
            True if point is in region
        """

        # Get region bounds in global coordinates
        global_x, global_y = self._get_global_coordinates(self.region.x, self.region.y)
        global_right = global_x + self.region.width
        global_bottom = global_y + self.region.height

        return global_x <= x < global_right and global_y <= y < global_bottom

    def get_center(self) -> Location:
        """Get center of region in global screen coordinates.

        Returns:
            Center location in global coordinates
        """
        center = self.region.center
        global_x, global_y = self._get_global_coordinates(center.x, center.y)
        return Location(global_x, global_y)

    def click(self) -> ActionResult:
        """Click in the center of this region.

        Coordinates are translated to global screen coordinates based on
        the associated monitor.

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

        Coordinates are translated to global screen coordinates based on
        the associated monitor.

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
        """Get the region in global screen coordinates for searching.

        Returns:
            The region with global coordinates
        """
        # Translate region to global coordinates
        global_x, global_y = self._get_global_coordinates(self.region.x, self.region.y)

        if global_x == self.region.x and global_y == self.region.y:
            # No translation needed
            return self.region

        # Return a new region with global coordinates
        return Region(
            x=global_x,
            y=global_y,
            width=self.region.width,
            height=self.region.height,
        )

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
        """Get the search region in global screen coordinates.

        Returns:
            The region with global coordinates
        """
        return self.search_region

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
