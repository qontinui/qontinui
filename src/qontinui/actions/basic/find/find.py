"""Find action - ported from Qontinui framework.

Core pattern matching action that locates GUI elements on the screen.

MIGRATION NOTE: This class now delegates to FindAction for actual pattern matching.
The ActionInterface pattern is preserved for compatibility with the Brobot-style model.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

from ....actions.find import FindAction
from ....actions.find import FindOptions as NewFindOptions
from ...action_interface import ActionInterface
from ...action_result import ActionResult
from ...action_type import ActionType
from ...object_collection import ObjectCollection
from .base_find_options import BaseFindOptions


class Find(ActionInterface):
    """Core pattern matching action that locates GUI elements on the screen.

    Port of Find from Qontinui framework class.

    Find is the fundamental action in Qontinui's visual GUI automation, implementing various
    pattern matching strategies to locate GUI elements. It embodies the visual recognition
    capability that enables the framework to interact with any GUI regardless of the underlying
    technology.

    Find strategies supported:
    - FIRST: Returns the first match found, optimized for speed
    - BEST: Returns the highest-scoring match from all possibilities
    - EACH: Returns one match per StateImage/Pattern
    - ALL: Returns all matches found, useful for lists and grids
    - CUSTOM: User-defined find strategies for special cases

    Advanced features:
    - Multi-pattern matching with StateImages containing multiple templates
    - Color-based matching using k-means profiles
    - Text extraction from matched regions (OCR integration)
    - Match fusion for combining overlapping results
    - Dynamic offset adjustments for precise targeting

    Find operations also handle non-image objects in ObjectCollections:
    - Existing Matches can be reused without re-searching
    - Regions are converted to matches for consistent handling
    - Locations provide direct targeting without pattern matching

    In the model-based approach, Find operations are context-aware through integration
    with StateMemory, automatically adjusting active states based on what is found. This
    enables the framework to maintain an accurate understanding of the current GUI state.
    """

    def __init__(
        self,
        find_pipeline: Optional["FindPipeline"] = None,
        find_action: FindAction | None = None,
    ) -> None:
        """Initialize Find action.

        Args:
            find_pipeline: The pipeline that orchestrates the find process (legacy)
            find_action: The FindAction for actual pattern matching
        """
        self.find_pipeline = find_pipeline
        self.find_action = find_action or FindAction()

    def get_action_type(self) -> ActionType:
        """Return the action type.

        Returns:
            ActionType.FIND
        """
        return ActionType.FIND

    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Execute the find operation to locate GUI elements on screen.

        This method now delegates to FindAction for actual pattern matching.

        When called directly (rather than through Action.perform), certain lifecycle
        operations are bypassed to avoid redundant processing:
        - Wait.pauseBeforeBegin - Pre-action delays
        - Matches.setSuccess - Success flag setting
        - Matches.setDuration - Timing measurements
        - Matches.saveSnapshots - Screenshot capturing
        - Wait.pauseAfterEnd - Post-action delays

        Args:
            matches: The ActionResult to populate with found matches. Must contain
                    valid BaseFindOptions configuration.
            object_collections: The collections containing patterns, regions, and other
                              objects to find. At least one collection must be provided.

        Raises:
            ValueError: If matches does not contain BaseFindOptions configuration
        """
        # Validate configuration
        action_config = matches.action_config
        if not isinstance(action_config, BaseFindOptions):
            raise ValueError("Find requires BaseFindOptions configuration")

        find_options_config = action_config

        # Use FindAction for pattern matching
        for obj_coll in object_collections:
            for state_image in obj_coll.state_images:
                pattern = state_image.get_pattern()
                if pattern:
                    options = NewFindOptions(
                        similarity=find_options_config.similarity,
                        find_all=True,
                    )
                    result = self.find_action.find(pattern, options)

                    if result.found:
                        for match in result.matches:
                            matches.add_match(match)  # type: ignore[attr-defined]

        # Set success based on whether matches were found
        if matches.matches:
            object.__setattr__(matches, "success", True)
        else:
            object.__setattr__(matches, "success", False)


class FindPipeline:
    """Placeholder for FindPipeline class.

    Orchestrates the entire find process including pattern matching,
    state management, match fusion, and post-processing.
    """

    def execute(
        self,
        find_options: BaseFindOptions,
        matches: ActionResult,
        object_collections: tuple[Any, ...],
    ) -> None:
        """Execute the find pipeline.

        Args:
            find_options: Configuration for the find operation
            matches: Result container to populate
            object_collections: Objects to find
        """
        # Placeholder implementation
        logger.debug("Executing find with strategy: %s", find_options.get_find_strategy())
        object.__setattr__(matches, "success", False)
