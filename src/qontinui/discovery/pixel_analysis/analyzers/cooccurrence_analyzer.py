"""Analyze StateImage co-occurrence patterns to discover states."""

import logging
from typing import Any

import numpy as np

from ...models import DiscoveredState, StateImage

logger = logging.getLogger(__name__)


class CooccurrenceAnalyzer:
    """Groups StateImages into states based on co-occurrence patterns."""

    def analyze(
        self, state_images: list[StateImage], screenshots: list[np.ndarray[Any, Any]]
    ) -> list[DiscoveredState]:
        """
        Group StateImages into states based on co-occurrence patterns.

        Args:
            state_images: List of discovered StateImages
            screenshots: Original screenshots

        Returns:
            List of discovered states
        """
        states = []

        # Build co-occurrence matrix
        n = len(state_images)
        cooccurrence: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.zeros(
            (n, n)
        )

        for i in range(n):
            for j in range(i, n):
                # Count screenshots where both appear
                common = set(state_images[i].screenshot_ids) & set(
                    state_images[j].screenshot_ids
                )
                cooccurrence[i, j] = len(common)
                cooccurrence[j, i] = len(common)

        # Normalize by total screenshots
        cooccurrence = (
            (cooccurrence / len(screenshots)).astype(np.float64).reshape((n, n))
        )

        # Group StateImages that ALWAYS appear together (when one appears, all appear)
        grouped = []
        used = set()

        for i in range(n):
            if i in used:
                continue

            group = [i]
            used.add(i)
            group_screenshots = set(state_images[i].screenshot_ids)

            for j in range(i + 1, n):
                if j in used:
                    continue

                j_screenshots = set(state_images[j].screenshot_ids)

                # Check if j always appears with the group
                # This means: wherever j appears, all group members appear too
                # AND wherever any group member appears, j appears too
                # In other words: they appear in exactly the same screenshots
                if j_screenshots == group_screenshots:
                    group.append(j)
                    used.add(j)

            if group:
                grouped.append(group)

        # Create states from groups
        for i, group_indices in enumerate(grouped):
            state_image_ids = [state_images[idx].id for idx in group_indices]

            # Find screenshots where ALL state images appear together (intersection)
            screenshot_sets = [
                set(state_images[idx].screenshot_ids) for idx in group_indices
            ]
            # Since we grouped images that appear in exactly the same screenshots,
            # all sets should be identical, so intersection equals any individual set
            if screenshot_sets:
                common_screenshots = sorted(screenshot_sets[0])
            else:
                common_screenshots = []

            state = DiscoveredState(
                id=f"state_{i:03d}",
                name=f"State_{i:03d}",
                state_image_ids=state_image_ids,
                screenshot_ids=common_screenshots,
                confidence=0.9,
            )
            states.append(state)

        return states
