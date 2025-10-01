"""Location - ported from Qontinui framework.

Represents a point on the screen in the framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .position import Position

if TYPE_CHECKING:
    from .region import Region


@dataclass
class Location:
    """Represents a point on the screen for action targets.

    Port of Location from Qontinui framework class.

    IMPORTANT DESIGN NOTE:
    ======================
    In Qontinui, Location and Anchor serve DIFFERENT purposes:
    - Location: Used for ACTION TARGETS (where to click, type, hover, etc.)
    - Anchor: Used for REGION DEFINITION and spatial relationships

    This differs from Brobot where the relationship may be structured differently.
    See LOCATION_ANCHOR_DESIGN.md for full design rationale.

    POSITIONING MODES:
    ==================
    A Location can be defined in two ways:
    1. ABSOLUTE: Using x,y pixel coordinates directly
       - Used when: fixed=True OR region=None
       - Example: Location(x=100, y=200, fixed=True)

    2. RELATIVE: As a percentage position within a Region
       - Used when: region is set AND fixed=False
       - Example: Location(region=match.region, position=Position.CENTER)
       - The region can be from: StateImage match, StateRegion, or any Region

    RESOLUTION LOGIC:
    =================
    The system determines positioning mode as follows:
    1. If fixed=True → Always use absolute (x, y)
    2. If region is set AND position is set → Use relative positioning
    3. Otherwise → Use absolute (x, y)

    Both modes support offset_x and offset_y for fine adjustments.

    RUNTIME BEHAVIOR:
    =================
    During execution, a Location linked to an image:
    1. Waits for the image to be found
    2. Sets its region to the match.region
    3. Calculates final position within that region
    4. Applies any offsets

    This allows Locations to adapt dynamically to where elements appear.

    KEY DIFFERENCES FROM ANCHOR:
    ============================
    - Location DOES NOT define region boundaries
    - Location IS NOT a component of region definition
    - Location IS the target for all action operations
    - Anchor IS for defining how regions relate to each other

    When you need to define a region's boundaries, use Anchors.
    When you need to click/type somewhere, use Location.
    """

    x: int = 0
    """X coordinate in pixels."""

    y: int = 0
    """Y coordinate in pixels."""

    name: str | None = None
    """Optional name for this location."""

    region: Region | None = None
    """Optional region for relative positioning. Can be set at runtime from a match."""

    position: Position | None = None
    """Position within region (as percentages)."""

    anchor: str | None = None
    """Reference to a named Anchor (for spatial relationships, not for this Location's position)."""

    offset_x: int = 0
    """X offset in pixels added to final position."""

    offset_y: int = 0
    """Y offset in pixels added to final position."""

    fixed: bool = False
    """If True, always use absolute coordinates (x,y) regardless of region.
    If False, use relative positioning when region is available."""

    reference_image_id: str | None = None
    """ID of StateImage this location is relative to. Used to link to match at runtime."""

    def __post_init__(self):
        """Validate location after initialization."""
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0

    @classmethod
    def from_tuple(cls, coords: tuple[int, int]) -> Location:
        """Create Location from tuple.

        Args:
            coords: (x, y) tuple

        Returns:
            New Location instance
        """
        return cls(x=coords[0], y=coords[1])

    def to_tuple(self) -> tuple[int, int]:
        """Convert to tuple.

        Returns:
            (x, y) tuple
        """
        return (self.x, self.y)

    def get_final_location(self) -> Location:
        """Get final location after applying region and offsets.

        Resolution logic:
        1. If fixed=True, always use absolute coordinates
        2. If region and position are set, use relative positioning
        3. Otherwise, use absolute coordinates

        This method is called at runtime to resolve the actual screen position.

        Returns:
            Final computed location with resolved coordinates
        """
        # Fixed positioning - always use absolute
        if self.fixed:
            return Location(
                x=self.x + self.offset_x, y=self.y + self.offset_y, name=self.name, fixed=True
            )

        # Relative positioning - calculate from region
        if self.region and self.position:
            base_x = self.region.x + int(self.region.width * self.position.percent_w)
            base_y = self.region.y + int(self.region.height * self.position.percent_h)
            return Location(
                x=base_x + self.offset_x,
                y=base_y + self.offset_y,
                name=self.name,
                fixed=False,
                reference_image_id=self.reference_image_id,
            )

        # Default to absolute coordinates
        return Location(
            x=self.x + self.offset_x, y=self.y + self.offset_y, name=self.name, fixed=self.fixed
        )

    def is_defined_with_region(self) -> bool:
        """Check if this location is defined relative to a region.

        Returns:
            True if defined with region
        """
        return self.region is not None

    def distance_to(self, other: Location) -> float:
        """Calculate distance to another location.

        Args:
            other: Other location

        Returns:
            Euclidean distance
        """
        import math

        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def offset(self, dx: int, dy: int) -> Location:
        """Create new location with offset.

        Args:
            dx: X offset
            dy: Y offset

        Returns:
            New offset location
        """
        return Location(
            x=self.x + dx,
            y=self.y + dy,
            name=self.name,
            region=self.region,
            position=self.position,
            anchor=self.anchor,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
        )

    def __str__(self) -> str:
        """String representation."""
        if self.name:
            return f"Location({self.name} at {self.x},{self.y})"
        return f"Location({self.x},{self.y})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Location(x={self.x}, y={self.y}, name={self.name})"
