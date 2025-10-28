"""Find color implementation - ported from Qontinui framework.

Color-based pattern matching and scene classification.

Refactored to use strategy pattern for different color matching approaches.
"""

import logging
from dataclasses import dataclass

from ....action_result import ActionResult
from ....object_collection import ObjectCollection
from .find_color_orchestrator import FindColorOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class FindColor:
    """Color-based pattern matching implementation.

    Port of FindColor from Qontinui framework class.

    Uses strategy pattern to support different color matching approaches:
    - K-means clustering
    - Mean/standard deviation statistics
    - Multi-class classification

    Workflow:
    1. Acquire scenes (screenshot or provided images)
    2. Gather classification images (targets and context)
    3. Perform pixel-level color classification
    4. Extract contiguous regions as match candidates
    5. Filter and sort matches by size or score

    Delegates all logic to FindColorOrchestrator for cleaner separation.
    """

    def __post_init__(self) -> None:
        """Initialize the orchestrator."""
        self._orchestrator = FindColorOrchestrator()

    def find(self, matches: ActionResult, object_collections: list[ObjectCollection]) -> None:
        """Find matches based on color.

        Delegates to orchestrator for strategy selection and execution.

        Args:
            matches: ActionResult to populate with found matches
            object_collections: List of object collections containing targets and options
        """
        self._orchestrator.find(matches, object_collections)
