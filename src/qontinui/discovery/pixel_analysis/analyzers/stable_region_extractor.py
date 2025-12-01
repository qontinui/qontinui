"""Extract stable regions from stability maps."""

import hashlib
import logging
from typing import Any

import cv2
import numpy as np

from ...models import AnalysisConfig
from ..merge_components import merge_nearby_components

logger = logging.getLogger(__name__)


class StableRegionExtractor:
    """Extracts stable regions from stability maps using component analysis."""

    def __init__(self, config: AnalysisConfig) -> None:
        """Initialize with analysis configuration."""
        self.config = config

    def extract(
        self, stability_map: np.ndarray[Any, Any], reference_image: np.ndarray[Any, Any]
    ) -> list[dict[str, Any]]:
        """
        Extract stable regions from stability map using component merging.

        Args:
            stability_map: Binary stability map
            reference_image: Reference screenshot for pixel data

        Returns:
            List of stable regions with coordinates and pixel data
        """
        # Use component merging to reduce region count
        use_merging = getattr(self.config, "use_component_merging", True)
        merge_gap = getattr(self.config, "merge_gap", 8)
        min_pixels = getattr(self.config, "min_component_pixels", 50)

        if use_merging:
            logger.info("Using component merging to extract regions")
            regions = self._extract_merged_regions(
                stability_map, reference_image, merge_gap, min_pixels
            )
        else:
            logger.info("Using original connected components method")
            regions = self._extract_connected_components(stability_map, reference_image)

        # Optionally decompose complex regions
        if self.config.enable_rectangle_decomposition:
            regions = self._decompose_complex_regions(regions)

        return regions

    def _extract_merged_regions(
        self,
        stability_map: np.ndarray[Any, Any],
        reference_image: np.ndarray[Any, Any],
        merge_gap: int,
        min_pixels: int,
    ) -> list[dict[str, Any]]:
        """Extract regions using component merging."""
        # Get merged regions
        merged_regions = merge_nearby_components(
            stability_map,
            max_gap=merge_gap,
            min_pixels=min_pixels,
            min_region_size=self.config.min_region_size,
            max_region_size=self.config.max_region_size,
        )

        # Convert merged regions to standard format
        regions = []
        for merged in merged_regions:
            # Extract pixel data from reference image
            x, y, x2, y2 = merged["x"], merged["y"], merged["x2"], merged["y2"]
            pixel_data = reference_image[y : y2 + 1, x : x2 + 1].copy()

            # Calculate pixel hash
            pixel_hash = self._calculate_pixel_hash(pixel_data)

            regions.append(
                {
                    "x": x,
                    "y": y,
                    "x2": x2,
                    "y2": y2,
                    "width": merged["width"],
                    "height": merged["height"],
                    "pixel_data": pixel_data,
                    "pixel_hash": pixel_hash,
                    "mask": merged["mask"],
                }
            )

        logger.info(f"Extracted {len(regions)} merged regions")
        return regions

    def _extract_connected_components(
        self, stability_map: np.ndarray[Any, Any], reference_image: np.ndarray[Any, Any]
    ) -> list[dict[str, Any]]:
        """Extract regions using connected components."""
        regions = []

        # Find connected components
        num_labels, labels = cv2.connectedComponents(stability_map)
        logger.info(f"Found {num_labels - 1} connected components")

        for label_id in range(1, min(num_labels, 1000)):  # Limit to 1000 for safety
            # Create mask for this component
            mask = (labels == label_id).astype(np.uint8)

            # Find bounding box
            coords = np.column_stack(np.where(mask))
            if len(coords) == 0:
                continue

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            width = x_max - x_min + 1
            height = y_max - y_min + 1

            # Check size constraints
            if (
                width < self.config.min_region_size[0]
                or height < self.config.min_region_size[1]
                or width > self.config.max_region_size[0]
                or height > self.config.max_region_size[1]
            ):
                continue

            # Extract pixel data
            pixel_data = reference_image[y_min : y_max + 1, x_min : x_max + 1].copy()

            # Calculate pixel hash
            pixel_hash = self._calculate_pixel_hash(pixel_data)

            regions.append(
                {
                    "x": int(x_min),
                    "y": int(y_min),
                    "x2": int(x_max),
                    "y2": int(y_max),
                    "width": width,
                    "height": height,
                    "pixel_data": pixel_data,
                    "pixel_hash": pixel_hash,
                    "mask": mask[y_min : y_max + 1, x_min : x_max + 1],
                }
            )

        return regions

    def _calculate_pixel_hash(self, pixel_data: np.ndarray[Any, Any]) -> str:
        """Calculate hash of pixel data for comparison."""
        data_bytes = pixel_data.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()

    def _decompose_complex_regions(
        self, regions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Decompose complex regions into rectangles.

        This method identifies hollow regions (like window frames) and
        decomposes them into smaller rectangular components (borders).

        Args:
            regions: List of regions to potentially decompose

        Returns:
            List of regions with complex shapes decomposed
        """
        decomposed_regions = []

        for region in regions:
            # Check if this region should be decomposed
            if self._should_decompose_region(region):
                # Try to decompose into rectangles
                sub_regions = self._decompose_into_rectangles(region)
                if sub_regions:
                    logger.debug(
                        f"Decomposed region at ({region['x']}, {region['y']}) "
                        f"into {len(sub_regions)} rectangles"
                    )
                    decomposed_regions.extend(sub_regions)
                else:
                    # If decomposition fails, keep original
                    decomposed_regions.append(region)
            else:
                # Keep region as-is
                decomposed_regions.append(region)

        logger.info(
            f"Decomposition: {len(regions)} regions -> {len(decomposed_regions)} regions"
        )
        return decomposed_regions

    def _should_decompose_region(self, region: dict[str, Any]) -> bool:
        """
        Determine if a region should be decomposed.

        Candidates for decomposition:
        - Large regions with low mask density (hollow/frame-like)
        - Regions with aspect ratio suggesting frame structure

        Args:
            region: Region to evaluate

        Returns:
            True if region should be decomposed
        """
        width = region["width"]
        height = region["height"]
        mask = region.get("mask")

        # Only decompose reasonably large regions
        if width < 50 or height < 50:
            return False

        # Check mask density if available
        if mask is not None:
            mask_density = np.sum(mask > 0) / (width * height)
            # Hollow regions have low density (e.g., < 0.3)
            if mask_density < 0.3:
                return True

        # Check aspect ratio - very wide or tall regions might be borders
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 5.0:  # Very elongated
            return True

        return False

    def _decompose_into_rectangles(
        self, region: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Decompose a complex region into simpler rectangles.

        Uses contour analysis to find rectangular components.

        Args:
            region: Region to decompose

        Returns:
            List of decomposed rectangular regions, or empty list if decomposition fails
        """
        mask = region.get("mask")
        if mask is None:
            return []

        pixel_data = region["pixel_data"]
        x_offset = region["x"]
        y_offset = region["y"]

        try:
            # Find contours in the mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            sub_regions = []
            for contour in contours:
                # Get bounding rectangle for this contour
                cx, cy, cw, ch = cv2.boundingRect(contour)

                # Skip very small components
                if cw < 10 or ch < 10:
                    continue

                # Extract sub-region data
                sub_mask = mask[cy : cy + ch, cx : cx + cw]
                sub_pixel_data = pixel_data[cy : cy + ch, cx : cx + cw]

                # Calculate pixel hash for sub-region
                pixel_hash = self._calculate_pixel_hash(sub_pixel_data)

                sub_regions.append(
                    {
                        "x": x_offset + cx,
                        "y": y_offset + cy,
                        "x2": x_offset + cx + cw - 1,
                        "y2": y_offset + cy + ch - 1,
                        "width": cw,
                        "height": ch,
                        "pixel_data": sub_pixel_data,
                        "pixel_hash": pixel_hash,
                        "mask": sub_mask,
                    }
                )

            # Only return decomposed regions if we got meaningful results
            # (more than 1 component, each reasonably sized)
            if len(sub_regions) > 1:
                return sub_regions

        except Exception as e:
            logger.debug(f"Failed to decompose region: {e}")

        return []
