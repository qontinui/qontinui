"""Classify action - ported from Qontinui framework.

Main entry point for color-based classification and matching.
"""

import logging
from dataclasses import dataclass

from ....action_interface import ActionInterface
from ....action_result import ActionResult
from ....action_type import ActionType
from ....object_collection import ObjectCollection
from .find_color import FindColor

logger = logging.getLogger(__name__)


@dataclass
class Classify(ActionInterface):
    """Performs color-based classification and matching.

    Port of Classify from Qontinui framework action.

    Main entry point for color-based operations including:
    - K-means clustering for dominant color detection
    - Statistical color profiling (MU strategy)
    - Multi-class scene classification

    This action serves as the primary interface for color-based
    operations in the framework, delegating to FindColor for
    the actual implementation.
    """

    find_color: FindColor

    def get_action_type(self) -> ActionType:
        """Get action type.

        Returns:
            CLASSIFY action type
        """
        return ActionType.CLASSIFY

    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Perform scene classification using color-based analysis.

        Delegates to FindColor which handles the actual classification logic.
        The ActionResult will contain classified regions sorted by size
        when using CLASSIFY action type.

        ObjectCollections follow the standard pattern:
        - First: Target images (optional for pure classification)
        - Second: Context images for classification
        - Third+: Scenes to classify

        Classification differs from standard finding operations:
        - Classifies every pixel in the scene
        - Returns regions sorted by size (largest first)
        - Can operate without specific target images
        - Provides complete scene segmentation

        Args:
            matches: Accumulates classification results
            object_collections: Configures targets, context, and scenes
        """
        # Simply delegate to FindColor's find method
        # FindColor will handle the classification based on ActionType
        self.find_color.find(matches, list(object_collections))
