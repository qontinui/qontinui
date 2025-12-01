"""Color clustering utilities - ported from Qontinui framework.

K-means clustering and color analysis utilities.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from ....model.element.color import RGB

logger = logging.getLogger(__name__)


@dataclass
class ColorCluster:
    """Represents a color cluster from k-means analysis.

    Port of color from Qontinui framework clustering functionality.
    """

    center: RGB  # Cluster center color
    size: int  # Number of pixels in cluster
    percentage: float  # Percentage of total pixels

    def distance_to(self, color: RGB) -> float:
        """Calculate distance to another color.

        Args:
            color: Color to compare

        Returns:
            Euclidean distance in RGB space
        """
        return self.center.distance_to(color)


@dataclass
class ColorClusterAnalyzer:
    """Analyzes images using k-means color clustering.

    Port of k from Qontinui framework-means color analysis.

    Identifies dominant colors in images for matching and classification.
    """

    n_clusters: int = 3  # Default number of clusters
    random_state: int = 42  # For reproducible results

    def analyze_image(
        self, image: np.ndarray[Any, Any], n_clusters: int | None = None
    ) -> list[ColorCluster]:
        """Analyze image colors using k-means clustering.

        Args:
            image: Image to analyze (BGR format)
            n_clusters: Number of clusters (uses default if None)

        Returns:
            List of color clusters sorted by size
        """
        if n_clusters is None:
            n_clusters = self.n_clusters

        # Reshape image to list of pixels
        pixels = image.reshape((-1, 3))
        total_pixels = len(pixels)

        # Apply k-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters, random_state=self.random_state, n_init=10
        )
        labels = kmeans.fit_predict(pixels)

        # Build clusters
        clusters = []
        for i, center in enumerate(kmeans.cluster_centers_):
            # Count pixels in this cluster
            size = np.sum(labels == i)
            percentage = (size / total_pixels) * 100

            # Convert BGR to RGB
            rgb = RGB(int(center[2]), int(center[1]), int(center[0]))

            cluster = ColorCluster(center=rgb, size=size, percentage=percentage)
            clusters.append(cluster)

        # Sort by size (largest first)
        clusters.sort(key=lambda c: c.size, reverse=True)

        logger.debug(
            f"Found {len(clusters)} color clusters, "
            f"dominant color: {clusters[0].center} ({clusters[0].percentage:.1f}%)"
        )

        return clusters

    def find_dominant_colors(
        self, image: np.ndarray[Any, Any], top_n: int = 3
    ) -> list[RGB]:
        """Find the most dominant colors in an image.

        Args:
            image: Image to analyze
            top_n: Number of dominant colors to return

        Returns:
            List of dominant RGB colors
        """
        clusters = self.analyze_image(image, n_clusters=max(top_n, 3))
        return [cluster.center for cluster in clusters[:top_n]]

    def match_color_in_image(
        self, image: np.ndarray[Any, Any], target_color: RGB, tolerance: int = 30
    ) -> float:
        """Check how well a target color matches the image.

        Args:
            image: Image to check
            target_color: Color to find
            tolerance: Maximum distance for a match

        Returns:
            Match score (0.0 to 1.0)
        """
        clusters = self.analyze_image(image)

        # Find best matching cluster
        best_distance = float("inf")
        best_cluster = None

        for cluster in clusters:
            distance = cluster.distance_to(target_color)
            if distance < best_distance:
                best_distance = distance
                best_cluster = cluster

        if best_distance > tolerance or best_cluster is None:
            return 0.0

        # Calculate score based on distance and cluster size
        distance_score = 1.0 - (best_distance / tolerance)
        size_score = best_cluster.percentage / 100.0

        # Combined score
        return distance_score * 0.7 + size_score * 0.3
