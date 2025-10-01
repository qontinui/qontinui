"""Component merging for reducing region count in State Discovery."""

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def merge_nearby_components(
    stability_map: np.ndarray[Any, Any],
    max_gap: int = 8,
    min_pixels: int = 50,
    min_region_size: tuple[int, int] = (20, 20),
    max_region_size: tuple[int, int] = (500, 500),
) -> list[dict[str, Any]]:
    """
    Merge nearby stable components to form meaningful regions.

    This connects:
    - Individual letters into words
    - Icon fragments into complete icons
    - Border segments into frames

    Args:
        stability_map: Binary map where 1 = stable pixel
        max_gap: Maximum pixel gap to bridge between components
        min_pixels: Minimum pixels for a valid region
        min_region_size: Minimum (width, height) for regions
        max_region_size: Maximum (width, height) for regions

    Returns:
        List of region dictionaries with bbox, mask, and pixel data
    """
    if stability_map.size == 0:
        logger.warning("Empty stability map provided")
        return []

    # Convert to uint8 if needed
    if stability_map.dtype != np.uint8:
        stability_map = stability_map.astype(np.uint8)

    logger.info(f"Merging components with max_gap={max_gap}, min_pixels={min_pixels}")
    logger.info(
        f"Stability map shape: {stability_map.shape}, stable pixels: {np.sum(stability_map)}"
    )

    # Step 1: Create dilation kernel
    kernel = np.ones((max_gap, max_gap), np.uint8)

    # Step 2: Dilate to connect nearby pixels
    dilated = cv2.dilate(stability_map, kernel, iterations=1)

    logger.info(f"After dilation: {np.sum(dilated)} pixels")

    # Step 3: Find connected components on dilated map
    num_labels, labels = cv2.connectedComponents(dilated)

    logger.info(f"Found {num_labels - 1} connected components after dilation")

    # Step 4: Process each component
    merged_regions = []

    for label_id in range(1, num_labels):  # Skip background (0)
        # Get the dilated component mask
        dilated_mask = labels == label_id

        # Get original stable pixels within this dilated region
        original_mask = dilated_mask & (stability_map > 0)

        # Count original pixels
        pixel_count = np.sum(original_mask)

        if pixel_count < min_pixels:
            continue

        # Get bounding box
        coords = np.column_stack(np.where(original_mask))
        if len(coords) == 0:
            continue

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        width = x_max - x_min + 1
        height = y_max - y_min + 1

        # Check size constraints
        if width < min_region_size[0] or height < min_region_size[1]:
            continue
        if width > max_region_size[0] or height > max_region_size[1]:
            continue

        # Extract the mask for this region
        region_mask = original_mask[y_min : y_max + 1, x_min : x_max + 1]

        # Calculate mask density (how filled is the bounding box)
        mask_density = pixel_count / (width * height)

        merged_regions.append(
            {
                "x": int(x_min),
                "y": int(y_min),
                "x2": int(x_max),
                "y2": int(y_max),
                "width": width,
                "height": height,
                "mask": region_mask,
                "pixel_count": int(pixel_count),
                "mask_density": float(mask_density),
                "label_id": label_id,
            }
        )

    logger.info(f"After size filtering: {len(merged_regions)} valid merged regions")

    # Log some statistics
    if merged_regions:
        sizes = [(r["width"], r["height"]) for r in merged_regions]
        densities = [r["mask_density"] for r in merged_regions]
        logger.info(f"Region sizes range: {min(sizes)} to {max(sizes)}")
        logger.info(f"Mask densities range: {min(densities):.2f} to {max(densities):.2f}")

    return merged_regions


def visualize_merged_regions(
    image: np.ndarray[Any, Any], merged_regions: list[dict[str, Any]], original_components: int = 0
) -> np.ndarray[Any, Any]:
    """
    Visualize merged regions on an image for debugging.

    Args:
        image: Original image
        merged_regions: List of merged regions
        original_components: Number of components before merging

    Returns:
        Annotated image
    """
    vis = image.copy()

    # Draw each merged region
    colors = np.random.randint(0, 255, (len(merged_regions), 3))

    for idx, region in enumerate(merged_regions):
        x, y, x2, y2 = region["x"], region["y"], region["x2"], region["y2"]
        color = colors[idx].tolist()

        # Draw bounding box
        cv2.rectangle(vis, (x, y), (x2, y2), color, 2)

        # Add label with size
        label = f"{region['width']}x{region['height']}"
        cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Add summary text
    text = f"Merged: {len(merged_regions)} regions"
    if original_components > 0:
        text += f" (from {original_components} components)"

    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return vis


def test_merge_parameters(
    stability_map: np.ndarray[Any, Any],
    gap_values: list[int] | None = None,
    min_pixel_values: list[int] | None = None,
) -> dict[str, Any]:
    """
    Test different merge parameters to find optimal values.

    Args:
        stability_map: Binary stability map
        gap_values: List of max_gap values to test
        min_pixel_values: List of min_pixels values to test

    Returns:
        Dictionary with test results
    """
    if gap_values is None:
        gap_values = [5, 8, 10, 15]
    if min_pixel_values is None:
        min_pixel_values = [30, 50, 100]

    results = {}

    # Get baseline (no merging)
    num_original = cv2.connectedComponents(stability_map.astype(np.uint8))[0] - 1

    for gap in gap_values:
        for min_pix in min_pixel_values:
            regions = merge_nearby_components(stability_map, max_gap=gap, min_pixels=min_pix)

            key = f"gap_{gap}_minpix_{min_pix}"
            results[key] = {
                "gap": gap,
                "min_pixels": min_pix,
                "num_regions": len(regions),
                "reduction": f"{num_original} -> {len(regions)}",
                "reduction_pct": (1 - len(regions) / num_original) * 100 if num_original > 0 else 0,
            }

            logger.info(
                f"Parameters {key}: {results[key]['reduction']} "
                f"({results[key]['reduction_pct']:.1f}% reduction)"
            )

    return results
