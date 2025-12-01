"""K-means clustering color strategy.

Uses k-means to find dominant colors in target images and matches
them in the scene.
"""

import logging
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from .....model.element.color import RGB
from .....model.match.match import Match
from ..color_matcher import ColorMatcher
from .base_strategy import BaseColorStrategy

logger = logging.getLogger(__name__)


class KMeansStrategy(BaseColorStrategy):
    """K-means clustering strategy for color finding.

    Extracts dominant colors from target images using k-means clustering,
    then finds regions in the scene matching those colors.
    """

    def find_color_regions(
        self,
        scene: np.ndarray[Any, Any],
        target_images: list[np.ndarray[Any, Any]],
        options: Any,
    ) -> list[Match]:
        """Find color regions using k-means clustering.

        Args:
            scene: Scene image to search
            target_images: Target images containing desired colors
            options: Color finding options

        Returns:
            List of matches
        """
        matches = []

        if not target_images:
            logger.warning("No target images for k-means color finding")
            return []

        for target in target_images:
            # Get dominant colors from target using k-means
            target_colors = self._get_kmeans_colors(target, options.get_kmeans())

            # Find regions in scene matching these colors
            for color in target_colors:
                color_matches = ColorMatcher.find_color_regions(
                    scene, color, options.get_diameter(), tolerance=30
                )
                matches.extend(color_matches)

        return matches

    def _get_kmeans_colors(self, image: np.ndarray[Any, Any], n_clusters: int) -> list[RGB]:
        """Get dominant colors using k-means.

        Args:
            image: Input image in BGR format
            n_clusters: Number of clusters

        Returns:
            List of dominant RGB colors
        """
        # Reshape image to list of pixels
        pixels = image.reshape((-1, 3))

        # Apply k-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)

        # Get cluster centers as RGB colors
        colors = []
        for center in kmeans.cluster_centers_:
            rgb = RGB(int(center[2]), int(center[1]), int(center[0]))  # BGR to RGB
            colors.append(rgb)

        return colors
