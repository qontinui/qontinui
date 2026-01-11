"""Base locator class for vision verification.

Provides the abstract base class for all vision locators with
common functionality for matching, filtering, and chaining.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.assertions import (
    BoundingBox,
    LocatorType,
    VisionLocatorMatch,
)

if TYPE_CHECKING:
    from qontinui.vision.verification.config import VisionConfig


@dataclass
class LocatorMatch:
    """Internal representation of a locator match."""

    bounds: BoundingBox
    confidence: float
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        """Get center point of the match."""
        return (
            self.bounds.x + self.bounds.width // 2,
            self.bounds.y + self.bounds.height // 2,
        )

    @property
    def area(self) -> int:
        """Get area of the match."""
        return int(self.bounds.width * self.bounds.height)

    def to_schema(self, locator_type: LocatorType, index: int = 0) -> VisionLocatorMatch:
        """Convert to schema model.

        Args:
            locator_type: Type of locator that found this match.
            index: Index among multiple matches.

        Returns:
            VisionLocatorMatch schema instance.
        """
        return VisionLocatorMatch(
            bounds=self.bounds,
            confidence=self.confidence,
            center=self.center,
            text=self.text,
            locator_type=locator_type,
            match_index=index,
        )


class BaseLocator(ABC):
    """Abstract base class for vision locators.

    Locators are responsible for finding elements on screen using
    various detection methods (template matching, OCR, ML, etc.).

    Subclasses must implement:
    - locator_type property
    - _find_matches method
    """

    def __init__(
        self,
        value: str,
        config: "VisionConfig | None" = None,
        **options: Any,
    ) -> None:
        """Initialize locator.

        Args:
            value: Locator value (image path, text, etc.).
            config: Vision configuration.
            **options: Additional locator-specific options.
        """
        self._value = value
        self._config = config
        self._options = options

        # Filter options
        self._nth: int | None = options.get("nth")
        self._first: bool = options.get("first", False)
        self._last: bool = options.get("last", False)

        # Region constraint
        self._search_region: BoundingBox | None = options.get("region")

        # Parent locator for chaining
        self._parent: BaseLocator | None = None

    @property
    @abstractmethod
    def locator_type(self) -> LocatorType:
        """Get the locator type."""
        pass

    @property
    def value(self) -> str:
        """Get the locator value."""
        return self._value

    @abstractmethod
    async def _find_matches(
        self,
        screenshot: NDArray[np.uint8],
        region: BoundingBox | None = None,
    ) -> list[LocatorMatch]:
        """Find all matches in screenshot.

        Args:
            screenshot: Screenshot as numpy array (BGR format).
            region: Optional region to search within.

        Returns:
            List of matches found.
        """
        pass

    async def find_all(
        self,
        screenshot: NDArray[np.uint8],
    ) -> list[LocatorMatch]:
        """Find all matching elements.

        Args:
            screenshot: Screenshot as numpy array.

        Returns:
            List of all matches.
        """
        # Determine search region
        region = self._search_region

        # If chained to parent, use parent's bounds as region
        if self._parent is not None:
            parent_matches = await self._parent.find_all(screenshot)
            if not parent_matches:
                return []
            # Use first parent match as region
            region = parent_matches[0].bounds

        # Find matches
        matches = await self._find_matches(screenshot, region)

        # Apply filters
        matches = self._apply_filters(matches)

        return matches

    async def find(
        self,
        screenshot: NDArray[np.uint8],
    ) -> LocatorMatch | None:
        """Find first matching element.

        Args:
            screenshot: Screenshot as numpy array.

        Returns:
            First match or None if not found.
        """
        matches = await self.find_all(screenshot)
        return matches[0] if matches else None

    async def exists(
        self,
        screenshot: NDArray[np.uint8],
    ) -> bool:
        """Check if element exists.

        Args:
            screenshot: Screenshot as numpy array.

        Returns:
            True if element exists.
        """
        match = await self.find(screenshot)
        return match is not None

    async def count(
        self,
        screenshot: NDArray[np.uint8],
    ) -> int:
        """Count matching elements.

        Args:
            screenshot: Screenshot as numpy array.

        Returns:
            Number of matches.
        """
        matches = await self.find_all(screenshot)
        return len(matches)

    def _apply_filters(self, matches: list[LocatorMatch]) -> list[LocatorMatch]:
        """Apply filter options to matches.

        Args:
            matches: List of matches to filter.

        Returns:
            Filtered list of matches.
        """
        if not matches:
            return matches

        # Sort by confidence (highest first)
        matches = sorted(matches, key=lambda m: m.confidence, reverse=True)

        # Apply nth filter
        if self._nth is not None:
            if 0 <= self._nth < len(matches):
                return [matches[self._nth]]
            return []

        # Apply first filter
        if self._first:
            return [matches[0]]

        # Apply last filter
        if self._last:
            return [matches[-1]]

        return matches

    def inside(self, parent: "BaseLocator") -> Self:
        """Constrain search to inside parent element.

        Args:
            parent: Parent locator to search within.

        Returns:
            Self for chaining.
        """
        self._parent = parent
        return self

    def nth(self, index: int) -> Self:
        """Select nth match (0-indexed).

        Args:
            index: Match index.

        Returns:
            Self for chaining.
        """
        self._nth = index
        return self

    def first(self) -> Self:
        """Select first match only.

        Returns:
            Self for chaining.
        """
        self._first = True
        return self

    def last(self) -> Self:
        """Select last match only.

        Returns:
            Self for chaining.
        """
        self._last = True
        return self

    def with_region(self, region: BoundingBox) -> Self:
        """Constrain search to region.

        Args:
            region: Region to search within.

        Returns:
            Self for chaining.
        """
        self._search_region = region
        return self

    def _crop_to_region(
        self,
        screenshot: NDArray[np.uint8],
        region: BoundingBox,
    ) -> NDArray[np.uint8]:
        """Crop screenshot to region.

        Args:
            screenshot: Full screenshot.
            region: Region to crop to.

        Returns:
            Cropped screenshot.
        """
        return screenshot[
            region.y : region.y + region.height,
            region.x : region.x + region.width,
        ]

    def _adjust_match_for_region(
        self,
        match: LocatorMatch,
        region: BoundingBox,
    ) -> LocatorMatch:
        """Adjust match coordinates for region offset.

        Args:
            match: Match with coordinates relative to region.
            region: Region the match was found in.

        Returns:
            Match with absolute coordinates.
        """
        return LocatorMatch(
            bounds=BoundingBox(
                x=match.bounds.x + region.x,
                y=match.bounds.y + region.y,
                width=match.bounds.width,
                height=match.bounds.height,
            ),
            confidence=match.confidence,
            text=match.text,
            metadata=match.metadata,
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({self._value!r})"


__all__ = [
    "BaseLocator",
    "LocatorMatch",
]
