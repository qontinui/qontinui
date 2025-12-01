"""
Visibility tracker for web extraction.

Tracks which elements appear/disappear together across multiple
snapshots to group them into states.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from playwright.async_api import Page

logger = logging.getLogger(__name__)


@dataclass
class VisibilitySnapshot:
    """A snapshot of element visibility at a point in time."""

    timestamp: float
    url: str
    element_visibility: dict[str, bool]  # element_id -> is_visible
    trigger_action: str | None = None  # Action that caused this state


@dataclass
class VisibilityCluster:
    """A cluster of elements that appear/disappear together."""

    cluster_id: str
    element_ids: set[str]
    confidence: float
    representative_snapshot: int  # Index of snapshot where all elements are visible


class VisibilityTracker:
    """Tracks element visibility across multiple snapshots to identify states."""

    def __init__(self, correlation_threshold: float = 0.8):
        """
        Initialize visibility tracker.

        Args:
            correlation_threshold: Minimum correlation for elements to be
                                 considered part of the same state (0-1).
        """
        self.correlation_threshold = correlation_threshold
        self.snapshots: list[VisibilitySnapshot] = []
        self.element_ids: list[str] = []  # Ordered list of all known elements
        self._element_id_to_index: dict[str, int] = {}

    def reset(self) -> None:
        """Reset the tracker state."""
        self.snapshots = []
        self.element_ids = []
        self._element_id_to_index = {}

    async def record_snapshot(
        self,
        page: Page,
        element_selectors: dict[str, str],  # element_id -> selector
        trigger_action: str | None = None,
    ) -> VisibilitySnapshot:
        """
        Record a visibility snapshot of all tracked elements.

        Args:
            page: Playwright page to check visibility on.
            element_selectors: Mapping of element IDs to CSS selectors.
            trigger_action: Optional description of action that caused this state.

        Returns:
            The recorded visibility snapshot.
        """
        import time

        visibility: dict[str, bool] = {}

        for element_id, selector in element_selectors.items():
            # Register element if new
            if element_id not in self._element_id_to_index:
                self._element_id_to_index[element_id] = len(self.element_ids)
                self.element_ids.append(element_id)

            # Check visibility
            try:
                is_visible = await self._check_element_visible(page, selector)
                visibility[element_id] = is_visible
            except Exception as e:
                logger.debug(f"Error checking visibility for {element_id}: {e}")
                visibility[element_id] = False

        snapshot = VisibilitySnapshot(
            timestamp=time.time(),
            url=page.url,
            element_visibility=visibility,
            trigger_action=trigger_action,
        )
        self.snapshots.append(snapshot)

        logger.debug(
            f"Recorded snapshot {len(self.snapshots)}: "
            f"{sum(visibility.values())}/{len(visibility)} elements visible"
        )

        return snapshot

    async def _check_element_visible(self, page: Page, selector: str) -> bool:
        """Check if an element is visible on the page."""
        try:
            element = await page.query_selector(selector)
            if element is None:
                return False

            # Check if element is visible
            is_visible = await element.is_visible()
            if not is_visible:
                return False

            # Check bounding box
            bbox = await element.bounding_box()
            if bbox is None or bbox["width"] <= 0 or bbox["height"] <= 0:
                return False

            # Check if in viewport
            viewport = page.viewport_size
            if viewport:
                if (
                    bbox["x"] + bbox["width"] < 0
                    or bbox["y"] + bbox["height"] < 0
                    or bbox["x"] > viewport["width"]
                    or bbox["y"] > viewport["height"]
                ):
                    return False

            return True

        except Exception:
            return False

    def get_visibility_matrix(self) -> np.ndarray:
        """
        Build a visibility matrix from all snapshots.

        Returns:
            2D numpy array of shape (num_snapshots, num_elements).
            Values are 1.0 for visible, 0.0 for not visible.
        """
        if not self.snapshots or not self.element_ids:
            return np.array([])

        matrix = np.zeros((len(self.snapshots), len(self.element_ids)))

        for snapshot_idx, snapshot in enumerate(self.snapshots):
            for element_id, is_visible in snapshot.element_visibility.items():
                if element_id in self._element_id_to_index:
                    element_idx = self._element_id_to_index[element_id]
                    matrix[snapshot_idx, element_idx] = 1.0 if is_visible else 0.0

        return matrix

    def compute_correlation_matrix(self) -> np.ndarray:
        """
        Compute pairwise correlation between element visibility patterns.

        Returns:
            2D numpy array of shape (num_elements, num_elements).
            Values are correlation coefficients (-1 to 1).
        """
        visibility_matrix = self.get_visibility_matrix()
        if visibility_matrix.size == 0 or len(self.snapshots) < 2:
            return np.array([])

        # Transpose to get elements as rows
        element_matrix = visibility_matrix.T

        # Handle constant columns (always visible or never visible)
        std = np.std(element_matrix, axis=1)
        constant_mask = std < 1e-10

        # Compute correlation matrix
        n_elements = element_matrix.shape[0]
        corr_matrix = np.eye(n_elements)

        for i in range(n_elements):
            for j in range(i + 1, n_elements):
                if constant_mask[i] or constant_mask[j]:
                    # If both are always visible together or never visible together
                    if np.array_equal(element_matrix[i], element_matrix[j]):
                        corr_matrix[i, j] = corr_matrix[j, i] = 1.0
                    else:
                        corr_matrix[i, j] = corr_matrix[j, i] = 0.0
                else:
                    corr = np.corrcoef(element_matrix[i], element_matrix[j])[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                    corr_matrix[i, j] = corr_matrix[j, i] = corr

        return corr_matrix

    def compute_state_clusters(self) -> list[VisibilityCluster]:
        """
        Cluster elements that appear/disappear together.

        Uses correlation-based clustering to group elements with
        highly correlated visibility patterns.

        Returns:
            List of visibility clusters, each representing a potential state.
        """
        corr_matrix = self.compute_correlation_matrix()
        if corr_matrix.size == 0:
            return []

        n_elements = len(self.element_ids)
        assigned = [False] * n_elements
        clusters: list[VisibilityCluster] = []

        # Greedy clustering based on correlation threshold
        for i in range(n_elements):
            if assigned[i]:
                continue

            # Start new cluster with element i
            cluster_elements = {self.element_ids[i]}
            assigned[i] = True

            # Add elements with high correlation
            for j in range(i + 1, n_elements):
                if assigned[j]:
                    continue

                # Check if j correlates with all elements in cluster
                correlates_with_all = True
                for elem_id in cluster_elements:
                    elem_idx = self._element_id_to_index[elem_id]
                    if corr_matrix[elem_idx, j] < self.correlation_threshold:
                        correlates_with_all = False
                        break

                if correlates_with_all:
                    cluster_elements.add(self.element_ids[j])
                    assigned[j] = True

            # Find representative snapshot (where most elements are visible)
            visibility_matrix = self.get_visibility_matrix()
            cluster_indices = [self._element_id_to_index[e] for e in cluster_elements]
            visibility_scores = np.sum(visibility_matrix[:, cluster_indices], axis=1)
            representative_snapshot = int(np.argmax(visibility_scores))

            # Calculate confidence based on correlation strength
            if len(cluster_elements) > 1:
                cluster_correlations = []
                for elem_id in cluster_elements:
                    idx = self._element_id_to_index[elem_id]
                    for other_id in cluster_elements:
                        if other_id != elem_id:
                            other_idx = self._element_id_to_index[other_id]
                            cluster_correlations.append(corr_matrix[idx, other_idx])
                confidence = float(np.mean(cluster_correlations))
            else:
                confidence = 1.0

            cluster = VisibilityCluster(
                cluster_id=f"cluster_{len(clusters):03d}",
                element_ids=cluster_elements,
                confidence=confidence,
                representative_snapshot=representative_snapshot,
            )
            clusters.append(cluster)

        logger.info(
            f"Found {len(clusters)} visibility clusters from "
            f"{n_elements} elements across {len(self.snapshots)} snapshots"
        )

        return clusters

    def get_visibility_changes(self, snapshot_idx: int) -> tuple[set[str], set[str]]:
        """
        Get elements that appeared/disappeared compared to previous snapshot.

        Args:
            snapshot_idx: Index of snapshot to compare.

        Returns:
            Tuple of (appeared_elements, disappeared_elements).
        """
        if snapshot_idx <= 0 or snapshot_idx >= len(self.snapshots):
            return set(), set()

        prev_snapshot = self.snapshots[snapshot_idx - 1]
        curr_snapshot = self.snapshots[snapshot_idx]

        appeared = set()
        disappeared = set()

        all_elements = set(prev_snapshot.element_visibility.keys()) | set(
            curr_snapshot.element_visibility.keys()
        )

        for element_id in all_elements:
            prev_visible = prev_snapshot.element_visibility.get(element_id, False)
            curr_visible = curr_snapshot.element_visibility.get(element_id, False)

            if curr_visible and not prev_visible:
                appeared.add(element_id)
            elif prev_visible and not curr_visible:
                disappeared.add(element_id)

        return appeared, disappeared

    def get_transition_candidates(self) -> list[dict[str, Any]]:
        """
        Identify potential transitions based on visibility changes.

        Returns:
            List of transition candidates with appeared/disappeared states.
        """
        candidates = []

        for i in range(1, len(self.snapshots)):
            appeared, disappeared = self.get_visibility_changes(i)

            if appeared or disappeared:
                candidates.append(
                    {
                        "snapshot_index": i,
                        "trigger_action": self.snapshots[i].trigger_action,
                        "url_change": self.snapshots[i].url != self.snapshots[i - 1].url,
                        "appeared_elements": appeared,
                        "disappeared_elements": disappeared,
                    }
                )

        return candidates

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about tracked visibility data."""
        if not self.snapshots:
            return {
                "num_snapshots": 0,
                "num_elements": 0,
                "num_always_visible": 0,
                "num_never_visible": 0,
                "num_variable": 0,
            }

        visibility_matrix = self.get_visibility_matrix()
        visibility_sums = np.sum(visibility_matrix, axis=0)

        return {
            "num_snapshots": len(self.snapshots),
            "num_elements": len(self.element_ids),
            "num_always_visible": int(np.sum(visibility_sums == len(self.snapshots))),
            "num_never_visible": int(np.sum(visibility_sums == 0)),
            "num_variable": int(
                np.sum((visibility_sums > 0) & (visibility_sums < len(self.snapshots)))
            ),
        }
