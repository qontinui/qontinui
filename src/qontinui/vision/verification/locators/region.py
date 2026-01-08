"""Region-based locator using coordinates.

Provides direct coordinate-based element targeting without
visual detection, useful for known fixed regions.
"""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.assertions import BoundingBox, LocatorType

from qontinui.vision.verification.locators.base import BaseLocator, LocatorMatch

if TYPE_CHECKING:
    from qontinui.vision.verification.config import VisionConfig

logger = logging.getLogger(__name__)


class RegionLocator(BaseLocator):
    """Locator using direct coordinates.

    Targets a specific region on screen by coordinates.
    Useful for fixed UI elements whose position is known.

    Usage:
        # By coordinates
        locator = RegionLocator(x=100, y=200, width=150, height=50)
        match = await locator.find(screenshot)

        # By BoundingBox
        locator = RegionLocator.from_bounds(bounds)

        # By named region (requires environment)
        locator = RegionLocator.from_name("header", environment=env)
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        config: "VisionConfig | None" = None,
        name: str | None = None,
        **options: Any,
    ) -> None:
        """Initialize region locator.

        Args:
            x: X coordinate (left).
            y: Y coordinate (top).
            width: Width in pixels.
            height: Height in pixels.
            config: Vision configuration.
            name: Optional region name.
            **options: Additional options.
        """
        value = name or f"region({x},{y},{width},{height})"
        super().__init__(value, config, **options)

        self._bounds = BoundingBox(x=x, y=y, width=width, height=height)
        self._name = name

    @classmethod
    def from_bounds(
        cls,
        bounds: BoundingBox,
        config: "VisionConfig | None" = None,
        name: str | None = None,
        **options: Any,
    ) -> "RegionLocator":
        """Create locator from BoundingBox.

        Args:
            bounds: Bounding box defining the region.
            config: Vision configuration.
            name: Optional region name.
            **options: Additional options.

        Returns:
            RegionLocator instance.
        """
        return cls(
            x=bounds.x,
            y=bounds.y,
            width=bounds.width,
            height=bounds.height,
            config=config,
            name=name,
            **options,
        )

    @classmethod
    def from_center(
        cls,
        center_x: int,
        center_y: int,
        width: int,
        height: int,
        config: "VisionConfig | None" = None,
        **options: Any,
    ) -> "RegionLocator":
        """Create locator from center point and dimensions.

        Args:
            center_x: Center X coordinate.
            center_y: Center Y coordinate.
            width: Width in pixels.
            height: Height in pixels.
            config: Vision configuration.
            **options: Additional options.

        Returns:
            RegionLocator instance.
        """
        x = center_x - width // 2
        y = center_y - height // 2
        return cls(x=x, y=y, width=width, height=height, config=config, **options)

    @property
    def locator_type(self) -> LocatorType:
        """Get the locator type."""
        return LocatorType.REGION

    @property
    def bounds(self) -> BoundingBox:
        """Get the region bounds."""
        return self._bounds

    async def _find_matches(
        self,
        screenshot: NDArray[np.uint8],
        region: BoundingBox | None = None,
    ) -> list[LocatorMatch]:
        """Find the region match.

        Region locators always return exactly one match (the defined region)
        as long as it fits within the screenshot bounds.

        Args:
            screenshot: Screenshot to search.
            region: Optional parent region constraint.

        Returns:
            List with one match, or empty if region is out of bounds.
        """
        h, w = screenshot.shape[:2]

        # Get effective bounds
        bounds = self._bounds
        if region is not None:
            # Intersect with parent region
            x = max(bounds.x, region.x)
            y = max(bounds.y, region.y)
            x2 = min(bounds.x + bounds.width, region.x + region.width)
            y2 = min(bounds.y + bounds.height, region.y + region.height)

            if x2 <= x or y2 <= y:
                return []  # No intersection

            bounds = BoundingBox(x=x, y=y, width=x2 - x, height=y2 - y)

        # Check if region is within screenshot bounds
        if (
            bounds.x < 0
            or bounds.y < 0
            or bounds.x + bounds.width > w
            or bounds.y + bounds.height > h
        ):
            logger.warning(f"Region {bounds} is out of screenshot bounds ({w}x{h})")
            # Clip to screenshot bounds
            x = max(0, bounds.x)
            y = max(0, bounds.y)
            x2 = min(w, bounds.x + bounds.width)
            y2 = min(h, bounds.y + bounds.height)

            if x2 <= x or y2 <= y:
                return []

            bounds = BoundingBox(x=x, y=y, width=x2 - x, height=y2 - y)

        # Return the region as a match
        return [
            LocatorMatch(
                bounds=bounds,
                confidence=1.0,  # Region is exact
                metadata={"name": self._name} if self._name else {},
            )
        ]

    def expand(self, pixels: int) -> "RegionLocator":
        """Create expanded region.

        Args:
            pixels: Pixels to expand in all directions.

        Returns:
            New RegionLocator with expanded bounds.
        """
        return RegionLocator(
            x=self._bounds.x - pixels,
            y=self._bounds.y - pixels,
            width=self._bounds.width + pixels * 2,
            height=self._bounds.height + pixels * 2,
            config=self._config,
            name=self._name,
        )

    def shrink(self, pixels: int) -> "RegionLocator":
        """Create shrunk region.

        Args:
            pixels: Pixels to shrink from all edges.

        Returns:
            New RegionLocator with shrunk bounds.
        """
        return RegionLocator(
            x=self._bounds.x + pixels,
            y=self._bounds.y + pixels,
            width=max(1, self._bounds.width - pixels * 2),
            height=max(1, self._bounds.height - pixels * 2),
            config=self._config,
            name=self._name,
        )


__all__ = ["RegionLocator"]
