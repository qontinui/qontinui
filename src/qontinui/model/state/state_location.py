"""StateLocation - ported from Qontinui framework.

Locations associated with states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..element import Location
from .action_history import ActionHistory

if TYPE_CHECKING:
    from qontinui.actions.action_result import ActionResult
    from qontinui.model.state.state import State


@dataclass
class StateLocation:
    """Location associated with a state.

    Port of StateLocation from Qontinui framework class.
    Represents a specific point in a state.
    """

    location: Location
    name: str | None = None
    owner_state: State | None = None

    # Location properties
    _anchor: bool = False  # If true, used as anchor point
    _fixed: bool = True  # If true, location is fixed

    # Monitor association - which monitor(s) this location is on
    # Required for multi-monitor support. Must be a specific monitor (not -1/all).
    monitors: list[int] = field(default_factory=lambda: [0])

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Action history for integration testing
    action_history: ActionHistory = field(default_factory=ActionHistory)

    def __post_init__(self):
        """Initialize location name."""
        if self.name is None:
            self.name = f"Location_{self.location.x}_{self.location.y}"

    def _get_global_coordinates(self) -> tuple[int, int]:
        """Convert location coordinates to global screen coordinates.

        If the location has monitor associations, the coordinates are assumed
        to be relative to that monitor and are translated to global coordinates.

        Returns:
            (x, y) tuple in global screen coordinates
        """
        x, y = self.location.x, self.location.y

        if self.monitors and self.monitors[0] >= 0:
            try:
                from qontinui.monitor.monitor_manager import MonitorManager

                manager = MonitorManager()
                monitor_idx = self.monitors[0]

                # Always translate coordinates - monitor 0 may not be at (0,0)
                # This handles cases where monitor 0 is positioned to the right or below other monitors
                x, y = manager.to_global_coordinates(x, y, monitor_idx)
            except Exception:
                # Fall back to using coordinates as-is
                pass

        return (x, y)

    def click(self) -> ActionResult:
        """Click at this location.

        Coordinates are translated to global screen coordinates based on
        the associated monitor.

        Returns:
            ActionResult from click
        """
        from ..actions import Action
        from ..actions.basic.click.click_options import ClickOptionsBuilder

        x, y = self._get_global_coordinates()
        click_options = ClickOptionsBuilder().build()
        action = Action(click_options)
        result: ActionResult = action.click(x, y)
        return result

    def hover(self) -> ActionResult:
        """Move mouse to this location.

        Coordinates are translated to global screen coordinates based on
        the associated monitor.

        Returns:
            ActionResult from move
        """
        from ..actions import Action

        x, y = self._get_global_coordinates()
        action = Action()
        result: ActionResult = action.move(x, y)
        return result

    def distance_to(self, other: StateLocation) -> float:
        """Calculate distance to another location.

        Args:
            other: Other StateLocation

        Returns:
            Distance in pixels
        """
        return self.location.distance_to(other.location)

    def set_anchor(self, anchor: bool = True) -> StateLocation:
        """Set whether this is an anchor point (fluent).

        Args:
            anchor: True if anchor point

        Returns:
            Self for chaining
        """
        self._anchor = anchor
        return self

    def set_fixed(self, fixed: bool = True) -> StateLocation:
        """Set whether location is fixed (fluent).

        Args:
            fixed: True if fixed

        Returns:
            Self for chaining
        """
        self._fixed = fixed
        return self

    @property
    def is_anchor(self) -> bool:
        """Check if this is an anchor point."""
        return self._anchor

    @property
    def is_fixed(self) -> bool:
        """Check if location is fixed."""
        return self._fixed

    def get_name(self) -> str | None:
        """Get the location name.

        Returns:
            Location name or None
        """
        return self.name

    def get_owner_state_name(self) -> str:
        """Get the owner state name.

        Returns:
            Owner state name or empty string if no owner
        """
        if self.owner_state is None:
            return ""
        return self.owner_state.name if hasattr(self.owner_state, "name") else ""

    def set_times_acted_on(self, times: int) -> None:
        """Set times acted on count.

        Args:
            times: Number of times acted on

        Note:
            This is a legacy method for compatibility. ActionHistory now uses
            add_record() to track individual actions rather than a simple counter.
            This method does nothing in the current implementation.
        """
        # ActionHistory no longer uses a simple counter
        # Individual action records are tracked via add_record()
        pass

    def __str__(self) -> str:
        """String representation."""
        state_name = self.owner_state.name if self.owner_state else "None"
        return f"StateLocation('{self.name}' at {self.location} in state '{state_name}')"
