"""Image-based locator using template matching.

Provides element detection using OpenCV template matching with
optional multi-scale support for scale-invariant matching.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.assertions import BoundingBox, LocatorType

from qontinui.vision.verification.locators.base import BaseLocator, LocatorMatch

if TYPE_CHECKING:
    from qontinui.vision.verification.config import VisionConfig

logger = logging.getLogger(__name__)

# OpenCV template matching methods
CV_METHODS = {
    "TM_CCOEFF": cv2.TM_CCOEFF,
    "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
    "TM_CCORR": cv2.TM_CCORR,
    "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
    "TM_SQDIFF": cv2.TM_SQDIFF,
    "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
}


class ImageLocator(BaseLocator):
    """Locator using OpenCV template matching.

    Finds elements by matching a template image against the screenshot.
    Supports multi-scale matching for handling size variations.

    Usage:
        locator = ImageLocator("button.png", threshold=0.9)
        matches = await locator.find_all(screenshot)
    """

    def __init__(
        self,
        image_path: str | Path,
        config: "VisionConfig | None" = None,
        threshold: float | None = None,
        method: str = "TM_CCOEFF_NORMED",
        multi_scale: bool = False,
        scale_range: tuple[float, float] = (0.8, 1.2),
        scale_steps: int = 5,
        grayscale: bool = False,
        **options: Any,
    ) -> None:
        """Initialize image locator.

        Args:
            image_path: Path to template image.
            config: Vision configuration.
            threshold: Match threshold (0.0-1.0).
            method: OpenCV template matching method.
            multi_scale: Enable multi-scale matching.
            scale_range: Scale range for multi-scale (min, max).
            scale_steps: Number of scale steps.
            grayscale: Convert to grayscale before matching.
            **options: Additional options.
        """
        super().__init__(str(image_path), config, **options)

        self._image_path = Path(image_path)
        self._threshold = threshold
        self._method_name = method
        self._multi_scale = multi_scale
        self._scale_range = scale_range
        self._scale_steps = scale_steps
        self._grayscale = grayscale

        # Load template
        self._template: NDArray[np.uint8] | None = None
        self._template_loaded = False

    @property
    def locator_type(self) -> LocatorType:
        """Get the locator type."""
        return LocatorType.IMAGE

    def _load_template(self) -> NDArray[np.uint8]:
        """Load template image.

        Returns:
            Template image as numpy array.

        Raises:
            FileNotFoundError: If template file not found.
            ValueError: If template cannot be loaded.
        """
        if self._template is not None and self._template_loaded:
            return self._template

        if not self._image_path.exists():
            raise FileNotFoundError(f"Template image not found: {self._image_path}")

        template = cv2.imread(str(self._image_path))
        if template is None:
            raise ValueError(f"Failed to load template image: {self._image_path}")

        self._template = template.astype(np.uint8)
        self._template_loaded = True
        return self._template

    def _get_threshold(self) -> float:
        """Get the match threshold.

        Returns:
            Threshold value.
        """
        if self._threshold is not None:
            return self._threshold
        if self._config is not None:
            return self._config.detection.template_threshold
        return 0.8

    def _get_cv_method(self) -> int:
        """Get OpenCV template matching method.

        Returns:
            OpenCV method constant.
        """
        method = self._method_name
        if self._config is not None:
            method = self._config.detection.template_method

        return CV_METHODS.get(method, cv2.TM_CCOEFF_NORMED)

    async def _find_matches(
        self,
        screenshot: NDArray[np.uint8],
        region: BoundingBox | None = None,
    ) -> list[LocatorMatch]:
        """Find all template matches.

        Args:
            screenshot: Screenshot to search.
            region: Optional region constraint.

        Returns:
            List of matches.
        """
        template = self._load_template()
        threshold = self._get_threshold()
        method = self._get_cv_method()

        # Crop to region if specified
        search_area = screenshot
        offset_x, offset_y = 0, 0
        if region is not None:
            search_area = self._crop_to_region(screenshot, region)
            offset_x, offset_y = region.x, region.y

        # Convert to grayscale if needed
        if self._grayscale:
            if len(search_area.shape) == 3:
                search_area = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            if len(template.shape) == 3:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).astype(np.uint8)

        # Find matches
        if self._multi_scale:
            matches = self._find_multi_scale(search_area, template, threshold, method)
        else:
            matches = self._find_single_scale(search_area, template, threshold, method)

        # Adjust coordinates for region offset
        for match in matches:
            match.bounds.x += offset_x
            match.bounds.y += offset_y

        return matches

    def _find_single_scale(
        self,
        search_area: NDArray[np.uint8],
        template: NDArray[np.uint8],
        threshold: float,
        method: int,
    ) -> list[LocatorMatch]:
        """Find matches at single scale.

        Args:
            search_area: Area to search.
            template: Template image.
            threshold: Match threshold.
            method: OpenCV method.

        Returns:
            List of matches.
        """
        h, w = template.shape[:2]

        # Check if template fits in search area
        if h > search_area.shape[0] or w > search_area.shape[1]:
            return []

        # Perform template matching
        result = cv2.matchTemplate(search_area, template, method)

        # Handle methods where minimum is best match
        if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            # Convert to similarity score
            result = 1 - result
            if method == cv2.TM_SQDIFF:
                # Normalize
                result = (result - result.min()) / (result.max() - result.min() + 1e-8)

        # Find all matches above threshold
        matches = []
        locations = np.where(result >= threshold)

        # Group nearby matches (non-maximum suppression)
        if len(locations[0]) > 0:
            # Cast result to float64 and locations to proper tuple type
            result_f64 = result.astype(np.float64)
            locations_tuple = (locations[0], locations[1])
            matches = self._non_max_suppression(result_f64, locations_tuple, w, h, threshold)

        return matches

    def _find_multi_scale(
        self,
        search_area: NDArray[np.uint8],
        template: NDArray[np.uint8],
        threshold: float,
        method: int,
    ) -> list[LocatorMatch]:
        """Find matches across multiple scales.

        Args:
            search_area: Area to search.
            template: Template image.
            threshold: Match threshold.
            method: OpenCV method.

        Returns:
            List of matches.
        """
        best_matches: list[LocatorMatch] = []
        scale_min, scale_max = self._scale_range

        scales = np.linspace(scale_min, scale_max, self._scale_steps)

        for scale in scales:
            # Resize template
            scaled_w = int(template.shape[1] * scale)
            scaled_h = int(template.shape[0] * scale)

            if scaled_w < 10 or scaled_h < 10:
                continue
            if scaled_h > search_area.shape[0] or scaled_w > search_area.shape[1]:
                continue

            scaled_template = cv2.resize(
                template, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA
            ).astype(np.uint8)

            # Find matches at this scale
            matches = self._find_single_scale(search_area, scaled_template, threshold, method)

            # Add scale info to metadata
            for match in matches:
                match.metadata["scale"] = scale

            best_matches.extend(matches)

        # Remove duplicates from different scales
        return self._deduplicate_matches(best_matches)

    def _non_max_suppression(
        self,
        result: NDArray[np.float64],
        locations: tuple[NDArray[np.intp], NDArray[np.intp]],
        width: int,
        height: int,
        threshold: float,
    ) -> list[LocatorMatch]:
        """Apply non-maximum suppression to remove overlapping matches.

        Args:
            result: Template matching result.
            locations: Locations above threshold.
            width: Template width.
            height: Template height.
            threshold: Match threshold.

        Returns:
            Filtered list of matches.
        """
        matches: list[LocatorMatch] = []

        # Create list of (y, x, confidence) tuples
        candidates = [
            (locations[0][i], locations[1][i], result[locations[0][i], locations[1][i]])
            for i in range(len(locations[0]))
        ]

        # Sort by confidence (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Track which regions have been used
        used_regions: list[tuple[int, int, int, int]] = []

        for y, x, confidence in candidates:
            # Check overlap with existing matches
            is_duplicate = False
            for uy, ux, uw, uh in used_regions:
                # Check if centers are too close
                cx1, cy1 = x + width // 2, y + height // 2
                cx2, cy2 = ux + uw // 2, uy + uh // 2

                if abs(cx1 - cx2) < width // 2 and abs(cy1 - cy2) < height // 2:
                    is_duplicate = True
                    break

            if not is_duplicate:
                matches.append(
                    LocatorMatch(
                        bounds=BoundingBox(x=int(x), y=int(y), width=width, height=height),
                        confidence=float(confidence),
                    )
                )
                used_regions.append((y, x, width, height))

        return matches

    def _deduplicate_matches(
        self,
        matches: list[LocatorMatch],
        distance_threshold: int = 20,
    ) -> list[LocatorMatch]:
        """Remove duplicate matches from different scales.

        Args:
            matches: List of matches to deduplicate.
            distance_threshold: Maximum center distance for duplicates.

        Returns:
            Deduplicated list.
        """
        if not matches:
            return matches

        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)

        unique: list[LocatorMatch] = []
        for match in matches:
            is_duplicate = False
            for existing in unique:
                # Check center distance
                dx = abs(match.center[0] - existing.center[0])
                dy = abs(match.center[1] - existing.center[1])
                if dx < distance_threshold and dy < distance_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(match)

        return unique


__all__ = ["ImageLocator"]
