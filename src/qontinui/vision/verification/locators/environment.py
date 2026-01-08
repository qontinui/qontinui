"""Environment-aware locator using discovered GUI data.

Provides element detection using the discovered GUI environment
including semantic regions, element patterns, and learned visual states.
"""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.assertions import BoundingBox, LocatorType
from qontinui_schemas.testing.environment import GUIEnvironment

from qontinui.vision.verification.errors import EnvironmentNotLoadedError
from qontinui.vision.verification.locators.base import BaseLocator, LocatorMatch
from qontinui.vision.verification.locators.region import RegionLocator

if TYPE_CHECKING:
    from qontinui.vision.verification.config import VisionConfig

logger = logging.getLogger(__name__)


class EnvironmentLocator(BaseLocator):
    """Locator using discovered GUI environment data.

    Leverages the GUI environment model to find elements using:
    - Named layout regions (header, sidebar, main_content)
    - Semantic colors (error, success, warning)
    - Element patterns (button, input, card)
    - Visual states (enabled, disabled, focused)

    Usage:
        # Find by region name
        locator = EnvironmentLocator("header", environment=env)
        match = await locator.find(screenshot)

        # Find elements with semantic color
        locator = EnvironmentLocator("error_text", type="semantic_color", environment=env)

        # Find by element pattern
        locator = EnvironmentLocator("button", type="element_pattern", environment=env)
    """

    def __init__(
        self,
        name: str,
        environment: GUIEnvironment | None = None,
        config: "VisionConfig | None" = None,
        locator_subtype: str = "region",
        **options: Any,
    ) -> None:
        """Initialize environment locator.

        Args:
            name: Name to look up in environment (region name, color name, etc.).
            environment: GUI environment model.
            config: Vision configuration.
            locator_subtype: Type of lookup: 'region', 'semantic_color', 'element_pattern'.
            **options: Additional options.
        """
        super().__init__(name, config, **options)

        self._name = name
        self._environment = environment
        self._locator_subtype = locator_subtype

    @property
    def locator_type(self) -> LocatorType:
        """Get the locator type."""
        return LocatorType.ENVIRONMENT

    def _ensure_environment(self) -> GUIEnvironment:
        """Ensure environment is loaded.

        Returns:
            GUI environment model.

        Raises:
            EnvironmentNotLoadedError: If environment not loaded.
        """
        if self._environment is None:
            raise EnvironmentNotLoadedError(
                feature=f"EnvironmentLocator({self._name})",
                suggestion="Pass environment to constructor or load via config.",
            )
        return self._environment

    async def _find_matches(
        self,
        screenshot: NDArray[np.uint8],
        region: BoundingBox | None = None,
    ) -> list[LocatorMatch]:
        """Find matches using environment data.

        Args:
            screenshot: Screenshot to search.
            region: Optional region constraint.

        Returns:
            List of matches.
        """
        env = self._ensure_environment()

        if self._locator_subtype == "region":
            return await self._find_by_region(screenshot, env, region)
        elif self._locator_subtype == "semantic_color":
            return await self._find_by_semantic_color(screenshot, env, region)
        elif self._locator_subtype == "element_pattern":
            return await self._find_by_element_pattern(screenshot, env, region)
        else:
            logger.warning(f"Unknown locator subtype: {self._locator_subtype}")
            return []

    async def _find_by_region(
        self,
        screenshot: NDArray[np.uint8],
        env: GUIEnvironment,
        parent_region: BoundingBox | None,
    ) -> list[LocatorMatch]:
        """Find by named layout region.

        Args:
            screenshot: Screenshot to search.
            env: GUI environment.
            parent_region: Optional parent region.

        Returns:
            List of matches.
        """
        # Look up region by name
        if self._name not in env.layout.regions:
            logger.warning(f"Region '{self._name}' not found in environment")
            return []

        layout_region = env.layout.regions[self._name]

        # Create region locator and delegate
        region_locator = RegionLocator.from_bounds(
            layout_region.bounds,
            config=self._config,
            name=self._name,
        )

        if parent_region is not None:
            region_locator = region_locator.with_region(parent_region)

        return await region_locator.find_all(screenshot)

    async def _find_by_semantic_color(
        self,
        screenshot: NDArray[np.uint8],
        env: GUIEnvironment,
        parent_region: BoundingBox | None,
    ) -> list[LocatorMatch]:
        """Find elements with semantic color.

        Looks for text/elements near the discovered semantic color.

        Args:
            screenshot: Screenshot to search.
            env: GUI environment.
            parent_region: Optional parent region.

        Returns:
            List of matches.
        """
        # Get semantic color
        semantic = env.colors.semantic_colors
        color_hex = None

        color_map = {
            "error": semantic.error,
            "success": semantic.success,
            "warning": semantic.warning,
            "info": semantic.info,
            "accent": semantic.accent,
            "background": semantic.background,
            "text_primary": semantic.text_primary,
            "text_secondary": semantic.text_secondary,
            "border": semantic.border,
        }

        color_hex = color_map.get(self._name)
        if color_hex is None:
            logger.warning(f"Semantic color '{self._name}' not found")
            return []

        # Find regions with this color
        return self._find_color_regions(screenshot, color_hex, parent_region)

    def _find_color_regions(
        self,
        screenshot: NDArray[np.uint8],
        color_hex: str,
        parent_region: BoundingBox | None,
    ) -> list[LocatorMatch]:
        """Find regions containing the specified color.

        Args:
            screenshot: Screenshot to search.
            color_hex: Hex color to find.
            parent_region: Optional region constraint.

        Returns:
            List of matches where color is found.
        """
        import cv2

        # Parse hex color
        color_hex = color_hex.lstrip("#")
        r = int(color_hex[0:2], 16)
        g = int(color_hex[2:4], 16)
        b = int(color_hex[4:6], 16)

        # Get tolerance from config
        tolerance = 20
        if self._config is not None:
            tolerance = self._config.comparison.color_tolerance

        # Crop to region if specified
        search_area = screenshot
        offset_x, offset_y = 0, 0
        if parent_region is not None:
            search_area = screenshot[
                parent_region.y : parent_region.y + parent_region.height,
                parent_region.x : parent_region.x + parent_region.width,
            ]
            offset_x, offset_y = parent_region.x, parent_region.y

        # Create color mask
        lower = np.array([max(0, b - tolerance), max(0, g - tolerance), max(0, r - tolerance)])
        upper = np.array(
            [min(255, b + tolerance), min(255, g + tolerance), min(255, r + tolerance)]
        )

        mask = cv2.inRange(search_area, lower, upper)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        matches = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Skip tiny regions
                continue

            x, y, w, h = cv2.boundingRect(contour)

            matches.append(
                LocatorMatch(
                    bounds=BoundingBox(
                        x=x + offset_x,
                        y=y + offset_y,
                        width=w,
                        height=h,
                    ),
                    confidence=0.9,  # Color match confidence
                    metadata={"color": f"#{r:02X}{g:02X}{b:02X}"},
                )
            )

        # Sort by area (largest first)
        matches.sort(key=lambda m: m.area, reverse=True)
        return matches

    async def _find_by_element_pattern(
        self,
        screenshot: NDArray[np.uint8],
        env: GUIEnvironment,
        parent_region: BoundingBox | None,
    ) -> list[LocatorMatch]:
        """Find elements matching a learned pattern.

        Args:
            screenshot: Screenshot to search.
            env: GUI environment.
            parent_region: Optional region constraint.

        Returns:
            List of matches.
        """
        # Look up element pattern
        if self._name not in env.element_patterns.patterns:
            logger.warning(f"Element pattern '{self._name}' not found")
            return []

        pattern = env.element_patterns.patterns[self._name]

        # If we have sample images, use template matching
        if pattern.examples:
            # Use first example as template
            first_example = pattern.examples[0]
            if first_example.image_path:
                from qontinui.vision.verification.locators.image import ImageLocator

                img_locator = ImageLocator(
                    first_example.image_path,
                    config=self._config,
                )
                if parent_region:
                    img_locator = img_locator.with_region(parent_region)
                return await img_locator.find_all(screenshot)

        # Otherwise use heuristic detection based on pattern characteristics
        return self._find_by_characteristics(screenshot, pattern, parent_region)

    def _find_by_characteristics(
        self,
        screenshot: NDArray[np.uint8],
        pattern: Any,
        parent_region: BoundingBox | None,
    ) -> list[LocatorMatch]:
        """Find elements by visual characteristics.

        Uses the learned pattern characteristics (size, shape, color)
        to detect similar elements.

        Args:
            screenshot: Screenshot to search.
            pattern: Element pattern with characteristics.
            parent_region: Optional region constraint.

        Returns:
            List of matches.
        """
        import cv2

        # Crop to region if specified
        search_area = screenshot
        offset_x, offset_y = 0, 0
        if parent_region is not None:
            search_area = screenshot[
                parent_region.y : parent_region.y + parent_region.height,
                parent_region.x : parent_region.x + parent_region.width,
            ]
            offset_x, offset_y = parent_region.x, parent_region.y

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        matches = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Check size constraints
            if pattern.typical_width:
                if w < pattern.typical_width.min or w > pattern.typical_width.max:
                    continue
            if pattern.typical_height:
                if h < pattern.typical_height.min or h > pattern.typical_height.max:
                    continue

            # Calculate confidence based on size match
            confidence = 0.7

            matches.append(
                LocatorMatch(
                    bounds=BoundingBox(
                        x=x + offset_x,
                        y=y + offset_y,
                        width=w,
                        height=h,
                    ),
                    confidence=confidence,
                    metadata={"element_type": self._name},
                )
            )

        return matches

    @classmethod
    def region(
        cls,
        name: str,
        environment: GUIEnvironment,
        config: "VisionConfig | None" = None,
    ) -> "EnvironmentLocator":
        """Create locator for named region.

        Args:
            name: Region name (header, sidebar, etc.).
            environment: GUI environment.
            config: Vision configuration.

        Returns:
            EnvironmentLocator instance.
        """
        return cls(name, environment=environment, config=config, locator_subtype="region")

    @classmethod
    def semantic_color(
        cls,
        name: str,
        environment: GUIEnvironment,
        config: "VisionConfig | None" = None,
    ) -> "EnvironmentLocator":
        """Create locator for semantic color regions.

        Args:
            name: Semantic color name (error, success, etc.).
            environment: GUI environment.
            config: Vision configuration.

        Returns:
            EnvironmentLocator instance.
        """
        return cls(name, environment=environment, config=config, locator_subtype="semantic_color")

    @classmethod
    def element_pattern(
        cls,
        name: str,
        environment: GUIEnvironment,
        config: "VisionConfig | None" = None,
    ) -> "EnvironmentLocator":
        """Create locator for element pattern.

        Args:
            name: Element pattern name (button, input, etc.).
            environment: GUI environment.
            config: Vision configuration.

        Returns:
            EnvironmentLocator instance.
        """
        return cls(name, environment=environment, config=config, locator_subtype="element_pattern")


__all__ = ["EnvironmentLocator"]
