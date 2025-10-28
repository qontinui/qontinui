"""Merging utilities for combining ActionResult instances.

Provides operations for merging matches and data from multiple ActionResult
instances into a single result.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .action_result import ActionResult


class ResultMerger:
    """Utility class for merging ActionResult instances.

    Provides methods for combining matches and non-match data from
    multiple results.
    """

    @staticmethod
    def add_match_objects(
        target: "ActionResult", source: "ActionResult"
    ) -> "ActionResult":
        """Merge match objects from source into target.

        Args:
            target: ActionResult to merge into
            source: ActionResult containing matches to add

        Returns:
            The target result (modified in place)
        """
        if source:
            for match in source.match_list:
                target.add(match)
        return target

    @staticmethod
    def add_non_match_results(
        target: "ActionResult", source: "ActionResult"
    ) -> "ActionResult":
        """Merge non-match data from source into target.

        This includes text, states, regions, movements, and execution history.

        Args:
            target: ActionResult to merge into
            source: ActionResult containing data to merge

        Returns:
            The target result (modified in place)
        """
        if source:
            if source.text:
                target.text = source.text
            if source.selected_text:
                target.selected_text = source.selected_text
            target.active_states.update(source.active_states)
            target.defined_regions.extend(source.defined_regions)
            target.movements.extend(source.movements)
            target.execution_history.extend(source.execution_history)
        return target

    @staticmethod
    def add_all_results(
        target: "ActionResult", source: "ActionResult"
    ) -> "ActionResult":
        """Merge all data from source into target.

        Combines both match objects and non-match data.

        Args:
            target: ActionResult to merge into
            source: ActionResult to merge completely

        Returns:
            The target result (modified in place)
        """
        if source:
            ResultMerger.add_match_objects(target, source)
            ResultMerger.add_non_match_results(target, source)
        return target
