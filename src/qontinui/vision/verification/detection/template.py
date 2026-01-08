"""Template matching detection engine with environment integration.

Provides template matching using OpenCV with integration of
learned element patterns from the GUI environment for improved
accuracy and reduced need for pre-stored templates.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.assertions import BoundingBox
from qontinui_schemas.testing.environment import (
    ElementPattern,
    ElementPatterns,
    GUIEnvironment,
)

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


@dataclass
class TemplateMatch:
    """Result from template matching."""

    bounds: BoundingBox
    confidence: float
    template_path: str | None = None
    template_hash: str | None = None
    scale: float = 1.0
    element_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        """Get center point of match."""
        return (
            self.bounds.x + self.bounds.width // 2,
            self.bounds.y + self.bounds.height // 2,
        )

    @property
    def area(self) -> int:
        """Get area of match."""
        return self.bounds.width * self.bounds.height


class TemplateEngine:
    """Template matching engine with environment-aware detection.

    Integrates with the discovered GUI environment to:
    - Use learned element patterns as templates
    - Apply color-based filtering using palette
    - Optimize search based on element characteristics
    - Support multi-scale matching

    Usage:
        engine = TemplateEngine(config, environment)
        matches = await engine.find_template("button.png", screenshot)
        matches = await engine.find_element_type("button", screenshot)
    """

    def __init__(
        self,
        config: "VisionConfig | None" = None,
        environment: GUIEnvironment | None = None,
    ) -> None:
        """Initialize template engine.

        Args:
            config: Vision configuration.
            environment: GUI environment with element patterns.
        """
        self._config = config
        self._environment = environment
        self._patterns: ElementPatterns | None = None

        if environment is not None:
            self._patterns = environment.element_patterns

        # Template cache
        self._template_cache: dict[str, NDArray[np.uint8]] = {}

    def set_environment(self, environment: GUIEnvironment) -> None:
        """Update the environment.

        Args:
            environment: New GUI environment.
        """
        self._environment = environment
        self._patterns = environment.element_patterns

    def _get_threshold(self) -> float:
        """Get match threshold from config.

        Returns:
            Match threshold (0.0-1.0).
        """
        if self._config is not None:
            return self._config.detection.template_threshold
        return 0.8

    def _get_method(self) -> int:
        """Get OpenCV template matching method.

        Returns:
            OpenCV method constant.
        """
        method_name = "TM_CCOEFF_NORMED"
        if self._config is not None:
            method_name = self._config.detection.template_method

        return CV_METHODS.get(method_name, cv2.TM_CCOEFF_NORMED)

    def _load_template(self, path: str | Path) -> NDArray[np.uint8]:
        """Load and cache template image.

        Args:
            path: Path to template image.

        Returns:
            Template image array.

        Raises:
            FileNotFoundError: If template not found.
            ValueError: If template cannot be loaded.
        """
        path_str = str(path)

        if path_str in self._template_cache:
            return self._template_cache[path_str]

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Template not found: {path}")

        template = cv2.imread(path_str)
        if template is None:
            raise ValueError(f"Failed to load template: {path}")

        self._template_cache[path_str] = template
        return template

    def _compute_template_hash(self, template: NDArray[np.uint8]) -> str:
        """Compute perceptual hash of template.

        Args:
            template: Template image.

        Returns:
            Hash string.
        """
        # Resize to small size for hashing
        small = cv2.resize(template, (16, 16))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Simple hash based on pixel values
        return hashlib.md5(gray.tobytes()).hexdigest()[:16]

    async def find_template(
        self,
        template_path: str | Path,
        image: NDArray[np.uint8],
        region: BoundingBox | None = None,
        threshold: float | None = None,
        multi_scale: bool = False,
        scale_range: tuple[float, float] = (0.8, 1.2),
        scale_steps: int = 5,
    ) -> list[TemplateMatch]:
        """Find template in image.

        Args:
            template_path: Path to template image.
            image: Image to search.
            region: Optional region to limit search.
            threshold: Match threshold (uses config default if None).
            multi_scale: Enable multi-scale matching.
            scale_range: Scale range for multi-scale (min, max).
            scale_steps: Number of scale steps.

        Returns:
            List of template matches.
        """
        template = self._load_template(template_path)
        template_hash = self._compute_template_hash(template)

        if threshold is None:
            threshold = self._get_threshold()

        # Crop to region if specified
        search_area = image
        offset_x, offset_y = 0, 0

        if region is not None:
            search_area = image[
                region.y : region.y + region.height,
                region.x : region.x + region.width,
            ]
            offset_x, offset_y = region.x, region.y

        # Find matches
        if multi_scale:
            matches = self._find_multi_scale(
                search_area, template, threshold, scale_range, scale_steps
            )
        else:
            matches = self._find_single_scale(search_area, template, threshold)

        # Adjust coordinates and add metadata
        for match in matches:
            match.bounds = BoundingBox(
                x=match.bounds.x + offset_x,
                y=match.bounds.y + offset_y,
                width=match.bounds.width,
                height=match.bounds.height,
            )
            match.template_path = str(template_path)
            match.template_hash = template_hash

        return matches

    def _find_single_scale(
        self,
        search_area: NDArray[np.uint8],
        template: NDArray[np.uint8],
        threshold: float,
    ) -> list[TemplateMatch]:
        """Find template at single scale.

        Args:
            search_area: Area to search.
            template: Template image.
            threshold: Match threshold.

        Returns:
            List of matches.
        """
        h, w = template.shape[:2]
        method = self._get_method()

        # Check if template fits
        if h > search_area.shape[0] or w > search_area.shape[1]:
            return []

        # Perform matching
        result = cv2.matchTemplate(search_area, template, method)

        # Handle methods where minimum is best
        if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            result = 1 - result
            if method == cv2.TM_SQDIFF:
                result = (result - result.min()) / (result.max() - result.min() + 1e-8)

        # Find matches above threshold
        locations = np.where(result >= threshold)
        if len(locations[0]) == 0:
            return []

        # Apply non-maximum suppression
        return self._non_max_suppression(result, locations, w, h, threshold)

    def _find_multi_scale(
        self,
        search_area: NDArray[np.uint8],
        template: NDArray[np.uint8],
        threshold: float,
        scale_range: tuple[float, float],
        scale_steps: int,
    ) -> list[TemplateMatch]:
        """Find template across multiple scales.

        Args:
            search_area: Area to search.
            template: Template image.
            threshold: Match threshold.
            scale_range: Scale range (min, max).
            scale_steps: Number of scale steps.

        Returns:
            List of matches from all scales.
        """
        all_matches: list[TemplateMatch] = []
        scales = np.linspace(scale_range[0], scale_range[1], scale_steps)

        for scale in scales:
            # Resize template
            new_w = int(template.shape[1] * scale)
            new_h = int(template.shape[0] * scale)

            if new_w < 10 or new_h < 10:
                continue
            if new_h > search_area.shape[0] or new_w > search_area.shape[1]:
                continue

            scaled_template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)

            matches = self._find_single_scale(search_area, scaled_template, threshold)

            # Add scale info
            for match in matches:
                match.scale = scale

            all_matches.extend(matches)

        # Deduplicate across scales
        return self._deduplicate_matches(all_matches)

    def _non_max_suppression(
        self,
        result: NDArray[np.float64],
        locations: tuple[NDArray[np.intp], NDArray[np.intp]],
        width: int,
        height: int,
        threshold: float,
    ) -> list[TemplateMatch]:
        """Apply non-maximum suppression.

        Args:
            result: Template matching result.
            locations: Locations above threshold.
            width: Template width.
            height: Template height.
            threshold: Match threshold.

        Returns:
            Filtered list of matches.
        """
        # Create candidates sorted by confidence
        candidates = [
            (locations[0][i], locations[1][i], result[locations[0][i], locations[1][i]])
            for i in range(len(locations[0]))
        ]
        candidates.sort(key=lambda x: x[2], reverse=True)

        matches: list[TemplateMatch] = []
        used_regions: list[tuple[int, int, int, int]] = []

        for y, x, confidence in candidates:
            # Check overlap with existing matches
            is_duplicate = False
            for uy, ux, uw, uh in used_regions:
                cx1, cy1 = x + width // 2, y + height // 2
                cx2, cy2 = ux + uw // 2, uy + uh // 2

                if abs(cx1 - cx2) < width // 2 and abs(cy1 - cy2) < height // 2:
                    is_duplicate = True
                    break

            if not is_duplicate:
                matches.append(
                    TemplateMatch(
                        bounds=BoundingBox(x=int(x), y=int(y), width=width, height=height),
                        confidence=float(confidence),
                    )
                )
                used_regions.append((y, x, width, height))

        return matches

    def _deduplicate_matches(
        self,
        matches: list[TemplateMatch],
        distance_threshold: int = 20,
    ) -> list[TemplateMatch]:
        """Remove duplicate matches from different scales.

        Args:
            matches: All matches.
            distance_threshold: Maximum center distance for duplicates.

        Returns:
            Deduplicated list.
        """
        if not matches:
            return matches

        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)

        unique: list[TemplateMatch] = []
        for match in matches:
            is_duplicate = False
            for existing in unique:
                dx = abs(match.center[0] - existing.center[0])
                dy = abs(match.center[1] - existing.center[1])
                if dx < distance_threshold and dy < distance_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(match)

        return unique

    async def find_element_type(
        self,
        element_type: str,
        image: NDArray[np.uint8],
        region: BoundingBox | None = None,
        threshold: float | None = None,
    ) -> list[TemplateMatch]:
        """Find elements of a learned type.

        Uses the discovered element patterns to find elements
        without requiring explicit template images.

        Args:
            element_type: Element type name (e.g., "button", "input").
            image: Image to search.
            region: Optional region to limit search.
            threshold: Match threshold.

        Returns:
            List of matches.
        """
        if self._patterns is None:
            logger.warning("No element patterns available")
            return []

        if element_type not in self._patterns.patterns:
            logger.warning(f"Unknown element type: {element_type}")
            return []

        pattern = self._patterns.patterns[element_type]

        # If we have sample images, use template matching
        if pattern.examples:
            all_matches: list[TemplateMatch] = []

            for example in pattern.examples:
                if example.image_path:
                    try:
                        matches = await self.find_template(
                            example.image_path,
                            image,
                            region=region,
                            threshold=threshold,
                        )
                        for match in matches:
                            match.element_type = element_type
                        all_matches.extend(matches)
                    except FileNotFoundError:
                        logger.warning(f"Example image not found: {example.image_path}")
                        continue

            return self._deduplicate_matches(all_matches)

        # Fall back to heuristic detection
        return self._find_by_characteristics(image, pattern, region, element_type)

    def _find_by_characteristics(
        self,
        image: NDArray[np.uint8],
        pattern: ElementPattern,
        region: BoundingBox | None,
        element_type: str,
    ) -> list[TemplateMatch]:
        """Find elements by visual characteristics.

        Uses learned pattern characteristics (size, shape, color)
        to detect elements without templates.

        Args:
            image: Image to search.
            pattern: Element pattern with characteristics.
            region: Optional region constraint.
            element_type: Element type name.

        Returns:
            List of matches.
        """
        # Crop to region
        search_area = image
        offset_x, offset_y = 0, 0

        if region is not None:
            search_area = image[
                region.y : region.y + region.height,
                region.x : region.x + region.width,
            ]
            offset_x, offset_y = region.x, region.y

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

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

            # Check aspect ratio for shape
            aspect_ratio = w / h if h > 0 else 0

            # Calculate confidence based on how well it matches pattern
            confidence = self._calculate_pattern_confidence(search_area, x, y, w, h, pattern)

            if confidence >= 0.5:  # Lower threshold for heuristic detection
                matches.append(
                    TemplateMatch(
                        bounds=BoundingBox(
                            x=x + offset_x,
                            y=y + offset_y,
                            width=w,
                            height=h,
                        ),
                        confidence=confidence,
                        element_type=element_type,
                        metadata={
                            "detection_method": "heuristic",
                            "aspect_ratio": aspect_ratio,
                        },
                    )
                )

        # Sort by confidence and limit results
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches[:20]  # Limit to top 20

    def _calculate_pattern_confidence(
        self,
        image: NDArray[np.uint8],
        x: int,
        y: int,
        w: int,
        h: int,
        pattern: ElementPattern,
    ) -> float:
        """Calculate confidence that region matches pattern.

        Args:
            image: Source image.
            x, y, w, h: Region bounds.
            pattern: Element pattern.

        Returns:
            Confidence score (0.0-1.0).
        """
        confidence = 0.5  # Base confidence

        # Extract region
        if y + h > image.shape[0] or x + w > image.shape[1]:
            return 0.0

        region = image[y : y + h, x : x + w]

        # Check color match
        if pattern.typical_colors:
            avg_color = region.mean(axis=(0, 1))  # BGR

            # Simple color distance check
            for color_hex in pattern.typical_colors:
                # Parse hex color
                c = color_hex.lstrip("#")
                r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

                # Calculate distance
                dist = abs(avg_color[2] - r) + abs(avg_color[1] - g) + abs(avg_color[0] - b)
                if dist < 100:  # Close color match
                    confidence += 0.2
                    break

        # Check corner radius (rounded vs sharp)
        if pattern.corner_radius is not None:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(gray, 4, 0.01, 10)

            if corners is not None:
                # More corners = sharper edges
                if pattern.corner_radius > 5 and len(corners) < 4:
                    confidence += 0.1  # Rounded matches
                elif pattern.corner_radius == 0 and len(corners) >= 4:
                    confidence += 0.1  # Sharp matches

        # Check shadow (blur at edges)
        if pattern.has_shadow:
            # Check for shadow by looking at edge blur
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()

            if variance < 500:  # Lower variance might indicate shadow blur
                confidence += 0.1

        return min(1.0, confidence)


# Global engine instance
_template_engine: TemplateEngine | None = None


def get_template_engine(
    config: "VisionConfig | None" = None,
    environment: GUIEnvironment | None = None,
) -> TemplateEngine:
    """Get the global template engine instance.

    Args:
        config: Optional vision configuration.
        environment: Optional GUI environment.

    Returns:
        TemplateEngine instance.
    """
    global _template_engine
    if _template_engine is None:
        _template_engine = TemplateEngine(config=config, environment=environment)
    elif environment is not None:
        _template_engine.set_environment(environment)
    return _template_engine


__all__ = [
    "TemplateEngine",
    "TemplateMatch",
    "get_template_engine",
]
