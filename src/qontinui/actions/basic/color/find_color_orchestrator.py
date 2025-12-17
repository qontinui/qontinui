"""Find color orchestrator.

Coordinates color finding operations by selecting and delegating
to appropriate color strategies.
"""

import logging
from typing import Any

import numpy as np

from ....action_result import ActionResult
from ....model.match.match import Match
from ....object_collection import ObjectCollection
from .color_find_options import ColorFindOptions, ColorStrategy
from .region_extractor import RegionExtractor
from .strategies import ClassificationStrategy, KMeansStrategy, MUStrategy

logger = logging.getLogger(__name__)


class FindColorOrchestrator:
    """Orchestrates color finding operations.

    Selects the appropriate strategy based on configuration and
    coordinates the overall color finding workflow.
    """

    def __init__(self) -> None:
        """Initialize orchestrator with strategies."""
        self._kmeans_strategy = KMeansStrategy()
        self._mu_strategy = MUStrategy()
        self._classification_strategy = ClassificationStrategy()

    def find(self, matches: ActionResult, object_collections: list[ObjectCollection]) -> None:
        """Find matches based on color.

        Args:
            matches: ActionResult to populate with found matches
            object_collections: List of object collections containing targets and options
        """
        if not object_collections:
            logger.warning("No object collections provided for color finding")
            return

        # Get options from first collection
        first_collection = object_collections[0]
        options = first_collection.action_options.find_options.color_find_options

        if options.get_diameter() < 0:
            return

        # Get scene to analyze
        scene = self._capture_scene(options)
        if scene is None:
            logger.warning("Could not capture scene for color finding")
            return

        # Get target images
        target_images = self._get_target_images(first_collection, options)
        if not target_images:
            logger.warning("No target images for color finding")
            return

        # Perform color-based matching using appropriate strategy
        found_matches = self._execute_strategy(scene, target_images, options)

        # Filter matches by area
        area_filtering = options.get_area_filtering()
        found_matches = RegionExtractor.filter_by_area(
            found_matches, area_filtering.min_area, area_filtering.max_area
        )

        # Sort matches
        found_matches = self._sort_matches(found_matches, options)

        # Limit number of matches
        max_matches = options.get_max_matches_to_act_on()
        if max_matches > 0:
            found_matches = found_matches[:max_matches]

        # Add matches to ActionResult
        for match in found_matches:
            matches.add_match(match)

        logger.debug(
            f"Found {len(found_matches)} color matches using {options.get_color_strategy()}"
        )

    def _execute_strategy(
        self,
        scene: np.ndarray[Any, Any],
        target_images: list[np.ndarray[Any, Any]],
        options: ColorFindOptions,
    ) -> list[Match]:
        """Execute the appropriate color finding strategy.

        Args:
            scene: Scene image to search
            target_images: Target images
            options: Color options

        Returns:
            List of matches
        """
        color_strategy = options.get_color_strategy()
        if color_strategy == ColorStrategy.KMEANS:
            return self._kmeans_strategy.find_color_regions(scene, target_images, options)
        elif color_strategy == ColorStrategy.MU:
            return self._mu_strategy.find_color_regions(scene, target_images, options)
        else:  # CLASSIFICATION
            return self._classification_strategy.find_color_regions(scene, target_images, options)

    def _sort_matches(self, matches: list[Match], options: ColorFindOptions) -> list[Match]:
        """Sort matches based on strategy.

        Args:
            matches: Matches to sort
            options: Color options

        Returns:
            Sorted matches
        """
        if options.get_color_strategy() == ColorStrategy.CLASSIFICATION:
            # Sort by region size (largest first)
            matches.sort(
                key=lambda m: m.region.width * m.region.height if m.region else 0,
                reverse=True,
            )
        else:
            # Sort by similarity score (highest first)
            matches.sort(key=lambda m: m.similarity, reverse=True)

        return matches

    def _get_target_images(
        self, object_collection: ObjectCollection, options: ColorFindOptions
    ) -> list[np.ndarray[Any, Any]]:
        """Get target images from collection.

        For classification strategy, includes both target and context images.
        For other strategies, only includes target images.

        Args:
            object_collection: Object collection
            options: Color options

        Returns:
            List of target images
        """
        # This would extract actual images from StateImages in collection
        # Classification strategy needs both target and context images
        if options.get_color_strategy() == ColorStrategy.CLASSIFICATION:
            return self._get_classification_images(object_collection)
        else:
            return self._get_basic_target_images(object_collection)

    def _get_basic_target_images(
        self, object_collection: ObjectCollection
    ) -> list[np.ndarray[Any, Any]]:
        """Get basic target images from collection.

        Args:
            object_collection: Object collection

        Returns:
            List of target images
        """
        # Placeholder - would extract actual images from StateImages
        return []

    def _get_classification_images(
        self, object_collection: ObjectCollection
    ) -> list[np.ndarray[Any, Any]]:
        """Get all images for classification (targets + context).

        Args:
            object_collection: Object collection

        Returns:
            List of classification images
        """
        # Placeholder - would extract all images including context
        return []

    def _capture_scene(self, options: ColorFindOptions) -> np.ndarray[Any, Any] | None:
        """Capture scene for analysis.

        Args:
            options: Color options

        Returns:
            Scene image or None
        """
        # Placeholder - would capture screenshot or use provided scene
        logger.debug("Capturing scene for color analysis")
        return None
