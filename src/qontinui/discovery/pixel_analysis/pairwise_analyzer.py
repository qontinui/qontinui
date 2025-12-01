"""Pairwise screenshot analysis for finding states in subsets."""

import hashlib
import logging
from collections import defaultdict
from typing import Any, cast

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PairwiseStateAnalyzer:
    """Analyzes screenshots pairwise to find states present in any subset."""

    def __init__(self, similarity_threshold: float = 0.95) -> None:
        self.similarity_threshold = similarity_threshold

    def analyze_screenshots(
        self,
        screenshots: list[np.ndarray[Any, Any]],
        min_region_size: tuple[int, int] = (20, 20),
        max_region_size: tuple[int, int] = (500, 500),
    ) -> dict[str, Any]:
        """
        Find all stable regions across any subset of screenshots.

        Args:
            screenshots: List of screenshot arrays
            min_region_size: Minimum (width, height) for valid regions
            max_region_size: Maximum (width, height) for valid regions

        Returns:
            Dictionary with regions grouped by their exact screenshot presence
        """
        n = len(screenshots)
        logger.info(f"Analyzing {n} screenshots pairwise ({n*(n-1)//2} pairs)")

        # Step 1: Find stable regions between each pair
        all_regions: dict[str, dict[str, Any]] = {}
        pair_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                pair_count += 1
                logger.debug(f"Comparing screenshots {i} and {j} (pair {pair_count})")

                # Find stable regions between this pair
                stable_regions = self._find_stable_between_pair(
                    screenshots[i], screenshots[j], min_region_size, max_region_size
                )

                # Add regions to collection
                for region in stable_regions:
                    region_hash = self._hash_region(region["pixels"])

                    if region_hash not in all_regions:
                        all_regions[region_hash] = {
                            "region": region,
                            "found_in_pairs": set(),
                            "present_in": set(),
                        }

                    # Record this pair
                    all_regions[region_hash]["found_in_pairs"].add((i, j))
                    # Initially assume present in both screenshots of the pair
                    all_regions[region_hash]["present_in"].add(i)
                    all_regions[region_hash]["present_in"].add(j)

        logger.info(f"Found {len(all_regions)} unique regions from pairwise comparison")

        # Step 2: Verify exact presence in ALL screenshots
        for _region_hash, region_data in all_regions.items():
            region = region_data["region"]
            exact_presence = set()

            for idx, screenshot in enumerate(screenshots):
                if self._is_region_present(region, screenshot):
                    exact_presence.add(idx)

            region_data["exact_presence"] = sorted(exact_presence)

        # Step 3: Group regions by exact screenshot presence to form states
        states = self._group_into_states(all_regions)

        return {
            "regions": all_regions,
            "states": states,
            "statistics": self._calculate_statistics(all_regions, states, n),
        }

    def _find_stable_between_pair(
        self,
        img1: np.ndarray[Any, Any],
        img2: np.ndarray[Any, Any],
        min_size: tuple[int, int],
        max_size: tuple[int, int],
    ) -> list[dict[str, Any]]:
        """Find regions that are stable between two screenshots."""

        # Calculate pixel-wise difference
        diff = cv2.absdiff(img1, img2)

        # Convert to grayscale if needed
        if len(diff.shape) == 3:
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        else:
            gray_diff = diff

        # Threshold to find stable pixels (low difference)
        max_diff = int(255 * (1 - self.similarity_threshold))
        stable_mask = (gray_diff < max_diff).astype(np.uint8)

        # Apply morphological operations to connect nearby pixels
        kernel = np.ones((5, 5), np.uint8)
        morphed_mask = cv2.morphologyEx(stable_mask, cv2.MORPH_CLOSE, kernel)
        stable_mask = morphed_mask.astype(np.uint8)

        # Find connected components
        num_labels, labels = cv2.connectedComponents(stable_mask)

        regions = []
        for label_id in range(1, num_labels):
            mask = labels == label_id

            # Get bounding box
            coords = np.column_stack(np.where(mask))
            if len(coords) < 50:  # Minimum pixel count
                continue

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            width = x_max - x_min + 1
            height = y_max - y_min + 1

            # Check size constraints
            if width < min_size[0] or height < min_size[1]:
                continue
            if width > max_size[0] or height > max_size[1]:
                continue

            # Extract region from first image
            region_pixels = img1[y_min : y_max + 1, x_min : x_max + 1].copy()
            region_mask = mask[y_min : y_max + 1, x_min : x_max + 1]

            regions.append(
                {
                    "bbox": (x_min, y_min, x_max, y_max),
                    "pixels": region_pixels,
                    "mask": region_mask,
                    "width": width,
                    "height": height,
                }
            )

        return regions

    def _is_region_present(
        self, region: dict[str, Any], screenshot: np.ndarray[Any, Any]
    ) -> bool:
        """Check if a region is present in a screenshot."""
        x_min, y_min, x_max, y_max = region["bbox"]

        # Check bounds
        if y_max >= screenshot.shape[0] or x_max >= screenshot.shape[1]:
            return False

        # Extract corresponding area
        roi = screenshot[y_min : y_max + 1, x_min : x_max + 1]

        if roi.shape != region["pixels"].shape:
            return False

        # Calculate similarity
        diff = cv2.absdiff(roi, region["pixels"])

        # Apply mask if available
        if "mask" in region and region["mask"] is not None:
            mask = region["mask"]
            masked_diff = (
                diff * np.expand_dims(mask, axis=2)
                if len(diff.shape) == 3
                else diff * mask
            )
            active_pixels = np.sum(mask)
            if active_pixels == 0:
                return False
            mean_diff = np.sum(masked_diff) / active_pixels
        else:
            mean_diff = np.mean(diff)

        similarity = 1.0 - (mean_diff / 255.0)
        return cast(bool, similarity >= self.similarity_threshold)

    def _hash_region(self, pixels: np.ndarray[Any, Any]) -> str:
        """Create hash for region deduplication."""
        return hashlib.sha256(pixels.tobytes()).hexdigest()

    def _group_into_states(
        self, regions: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Group regions by their exact screenshot presence."""

        # Group regions by their presence pattern
        presence_groups = defaultdict(list)

        for region_hash, region_data in regions.items():
            presence_key = tuple(region_data["exact_presence"])
            presence_groups[presence_key].append(region_hash)

        # Create states from groups
        states = []
        for state_id, (presence, region_hashes) in enumerate(presence_groups.items()):
            # Only create states with multiple regions or significant presence
            if len(region_hashes) >= 1 and len(presence) >= 2:
                states.append(
                    {
                        "id": f"state_{state_id:03d}",
                        "region_hashes": region_hashes,
                        "region_count": len(region_hashes),
                        "screenshot_ids": list(presence),
                        "screenshot_count": len(presence),
                    }
                )

        # Sort by number of screenshots (most common states first)
        states.sort(key=lambda s: s["screenshot_count"], reverse=True)  # type: ignore[arg-type, return-value]

        return states

    def _calculate_statistics(
        self,
        regions: dict[str, dict[str, Any]],
        states: list[dict[str, Any]],
        num_screenshots: int,
    ) -> dict[str, Any]:
        """Calculate analysis statistics."""

        # Region presence distribution
        presence_counts: defaultdict[int, int] = defaultdict(int)
        for region_data in regions.values():
            count = len(region_data["exact_presence"])
            presence_counts[count] += 1

        return {
            "total_screenshots": num_screenshots,
            "unique_regions": len(regions),
            "total_states": len(states),
            "pairs_analyzed": num_screenshots * (num_screenshots - 1) // 2,
            "presence_distribution": dict(presence_counts),
            "avg_regions_per_state": (
                sum(s["region_count"] for s in states) / len(states) if states else 0
            ),
        }


def find_states_in_subsets(
    screenshots: list[np.ndarray[Any, Any]], similarity_threshold: float = 0.95
) -> dict[str, Any]:
    """
    Convenience function to find states present in any subset of screenshots.

    Args:
        screenshots: List of screenshot arrays
        similarity_threshold: Threshold for region similarity

    Returns:
        Analysis results with regions and states
    """
    analyzer = PairwiseStateAnalyzer(similarity_threshold)
    return analyzer.analyze_screenshots(screenshots)
