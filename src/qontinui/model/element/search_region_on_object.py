"""SearchRegionOnObject class - faithful port from Brobot framework.

Configuration for deriving search regions from another state object's match.
"""


from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .match_adjustment_options import MatchAdjustmentOptions


class StateObjectType(Enum):
    """Type of state object - matches Brobot's StateObject.Type enum."""

    IMAGE = auto()
    REGION = auto()
    LOCATION = auto()
    STRING = auto()
    TEXT = auto()


@dataclass
class SearchRegionOnObject:
    """Configuration for deriving search regions from another state object's match.

    Faithful port of Brobot's SearchRegionOnObject class.

    This allows state objects to define their search areas dynamically based on the
    location of other objects, even from different states. For example, you can search
    for a button relative to a matched logo, or search within a dialog frame that was
    found dynamically.

    Example:
        # Search in the region 30 pixels below a matched menu item
        search_region = SearchRegionOnObject(
            target_state_name="MainMenu",
            target_object_name="FileMenu",
            target_type=StateObjectType.IMAGE,
            adjustments=MatchAdjustmentOptions(add_y=30, add_h=200)
        )
    """

    # Which type of state object to reference (IMAGE, REGION, LOCATION, etc.)
    target_type: StateObjectType | None = None

    # Name of the state containing the target object (None = same state)
    target_state_name: str | None = None

    # Name of the target object to use as the base region
    target_object_name: str | None = None

    # Optional adjustments to apply to the target object's region
    adjustments: MatchAdjustmentOptions | None = None

    def set_target_type(self, target_type: StateObjectType) -> SearchRegionOnObject:
        """Set the target object type.

        Args:
            target_type: Type of target object

        Returns:
            Self for fluent interface
        """
        self.target_type = target_type
        return self

    def set_target_state_name(self, state_name: str | None) -> SearchRegionOnObject:
        """Set the target state name.

        Args:
            state_name: Name of state containing target (None = same state)

        Returns:
            Self for fluent interface
        """
        self.target_state_name = state_name
        return self

    def set_target_object_name(self, object_name: str) -> SearchRegionOnObject:
        """Set the target object name.

        Args:
            object_name: Name of target object

        Returns:
            Self for fluent interface
        """
        self.target_object_name = object_name
        return self

    def set_adjustments(self, adjustments: MatchAdjustmentOptions) -> SearchRegionOnObject:
        """Set adjustment options.

        Args:
            adjustments: Adjustments to apply to target region

        Returns:
            Self for fluent interface
        """
        self.adjustments = adjustments
        return self

    def copy(self) -> SearchRegionOnObject:
        """Create a copy of this search region configuration.

        Returns:
            New SearchRegionOnObject instance
        """
        return SearchRegionOnObject(
            target_type=self.target_type,
            target_state_name=self.target_state_name,
            target_object_name=self.target_object_name,
            adjustments=self.adjustments,
        )

    def __str__(self) -> str:
        """String representation."""
        type_name = self.target_type.name if self.target_type else "None"
        return (
            f"SearchRegionOnObject(type={type_name}, "
            f"state={self.target_state_name}, "
            f"object={self.target_object_name})"
        )

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"SearchRegionOnObject(target_type={self.target_type}, "
            f"target_state_name='{self.target_state_name}', "
            f"target_object_name='{self.target_object_name}', "
            f"has_adjustments={self.adjustments is not None})"
        )
