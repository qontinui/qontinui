"""Find action - ported from Qontinui framework.

Core pattern matching action that locates GUI elements on the screen.
"""

from typing import Any, Optional

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

    def __init__(self, find_pipeline: Optional["FindPipeline"] = None) -> None:
        """Initialize Find action.

        Args:
            find_pipeline: The pipeline that orchestrates the find process
        """
        self.find_pipeline = find_pipeline

    def get_action_type(self) -> ActionType:
        """Return the action type.

        Returns:
            ActionType.FIND
        """
        return ActionType.FIND

    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Execute the find operation to locate GUI elements on screen.

        This method serves as a facade, delegating the entire find process to the
        FindPipeline. The pipeline handles all orchestration including pattern matching,
        state management, match fusion, and post-processing adjustments.

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

        find_options = action_config

        # Delegate entire orchestration to the pipeline
        if self.find_pipeline:
            self.find_pipeline.execute(find_options, matches, object_collections)
        else:
            # Placeholder implementation when pipeline not available
            print("FindPipeline not available, performing placeholder find")
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
        print(f"Executing find with strategy: {find_options.get_find_strategy()}")
        object.__setattr__(matches, "success", False)
