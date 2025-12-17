"""Declarative region definitions for applications.

Provides a way to define and manage regions declaratively,
similar to Brobot's approach.
"""

import logging
from collections.abc import Callable
from typing import Any

from ..model.element import Region
from .region_builder import RegionBuilder

logger = logging.getLogger(__name__)


class RegionDefinitions:
    """Container for declarative region definitions.

    Following Brobot principles:
    - Centralized region management
    - Named regions for easy reference
    - Support for dynamic regions
    - Lazy evaluation

    Example usage:
        regions = RegionDefinitions()

        # Define static regions
        regions.define("header", RegionBuilder()
            .at_screen_anchor(AnchorPoint.SCREEN_TOP_LEFT)
            .with_size(1920, 100)
            .build())

        # Define dynamic regions (evaluated when accessed)
        regions.define_dynamic("search_results", lambda: RegionBuilder()
            .below(regions.get("header"))
            .with_offset(0, 20)
            .with_size(800, 600)
            .build())
    """

    def __init__(self) -> None:
        """Initialize region definitions."""
        self._static_regions: dict[str, Region] = {}
        self._dynamic_regions: dict[str, Callable[[], Region]] = {}
        self._cached_dynamic: dict[str, Region] = {}

        # Define common screen regions
        self._define_screen_regions()

    def define(self, name: str, region: Region) -> "RegionDefinitions":
        """Define a static region.

        Args:
            name: Name for the region
            region: The region to store

        Returns:
            Self for chaining
        """
        self._static_regions[name] = region
        logger.debug(f"Defined static region '{name}': {region}")
        return self

    def define_builder(self, name: str, builder: RegionBuilder) -> "RegionDefinitions":
        """Define a region using a builder.

        Args:
            name: Name for the region
            builder: RegionBuilder to build the region

        Returns:
            Self for chaining
        """
        return self.define(name, builder.build())

    def define_dynamic(self, name: str, factory: Callable[[], Region]) -> "RegionDefinitions":
        """Define a dynamic region that's evaluated when accessed.

        Args:
            name: Name for the region
            factory: Function that returns the region

        Returns:
            Self for chaining
        """
        self._dynamic_regions[name] = factory
        # Clear cache if it exists
        if name in self._cached_dynamic:
            del self._cached_dynamic[name]
        logger.debug(f"Defined dynamic region '{name}'")
        return self

    def get(self, name: str, cache: bool = True) -> Region | None:
        """Get a region by name.

        Args:
            name: Region name
            cache: Whether to cache dynamic regions

        Returns:
            Region or None if not found
        """
        # Check static regions first
        if name in self._static_regions:
            return self._static_regions[name]

        # Check dynamic regions
        if name in self._dynamic_regions:
            # Check cache
            if cache and name in self._cached_dynamic:
                return self._cached_dynamic[name]

            # Evaluate dynamic region
            try:
                region = self._dynamic_regions[name]()
                if cache:
                    self._cached_dynamic[name] = region
                return region
            except Exception as e:
                logger.error(f"Error evaluating dynamic region '{name}': {e}")
                return None

        return None

    def update(self, name: str, region: Region) -> bool:
        """Update an existing region.

        Args:
            name: Region name
            region: New region value

        Returns:
            True if updated
        """
        if name in self._static_regions:
            self._static_regions[name] = region
            return True
        elif name in self._dynamic_regions:
            # Convert to static
            self._static_regions[name] = region
            del self._dynamic_regions[name]
            if name in self._cached_dynamic:
                del self._cached_dynamic[name]
            return True

        return False

    def remove(self, name: str) -> bool:
        """Remove a region definition.

        Args:
            name: Region name

        Returns:
            True if removed
        """
        removed = False

        if name in self._static_regions:
            del self._static_regions[name]
            removed = True

        if name in self._dynamic_regions:
            del self._dynamic_regions[name]
            removed = True

        if name in self._cached_dynamic:
            del self._cached_dynamic[name]

        return removed

    def clear_cache(self):
        """Clear cached dynamic regions."""
        self._cached_dynamic.clear()

    def get_all_names(self) -> list[Any]:
        """Get all defined region names.

        Returns:
            List of region names
        """
        return list(self._static_regions.keys()) + list(self._dynamic_regions.keys())

    def _define_screen_regions(self):
        """Define common screen regions."""
        screen = Region()  # Gets screen dimensions

        # Screen quadrants
        half_w = screen.width // 2
        half_h = screen.height // 2

        self.define("screen", screen)
        self.define("screen_top_half", Region(0, 0, screen.width, half_h))
        self.define("screen_bottom_half", Region(0, half_h, screen.width, half_h))
        self.define("screen_left_half", Region(0, 0, half_w, screen.height))
        self.define("screen_right_half", Region(half_w, 0, half_w, screen.height))

        # Quadrants
        self.define("screen_top_left", Region(0, 0, half_w, half_h))
        self.define("screen_top_right", Region(half_w, 0, half_w, half_h))
        self.define("screen_bottom_left", Region(0, half_h, half_w, half_h))
        self.define("screen_bottom_right", Region(half_w, half_h, half_w, half_h))

        # Common UI regions (can be overridden)
        self.define("header", Region(0, 0, screen.width, 100))
        self.define("footer", Region(0, screen.height - 100, screen.width, 100))
        self.define("sidebar_left", Region(0, 0, 250, screen.height))
        self.define("sidebar_right", Region(screen.width - 250, 0, 250, screen.height))
        self.define("content", Region(250, 100, screen.width - 500, screen.height - 200))
