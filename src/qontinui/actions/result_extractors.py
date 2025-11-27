"""Extraction utilities for ActionResult data.

Provides specialized methods for extracting matches, locations, and other
data from ActionResult instances.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..find.match import Match
    from ..model.element.location import Location
    from .action_result import ActionResult


class ResultExtractor:
    """Utility class for extracting data from ActionResult instances.

    Provides methods for finding best matches, extracting locations,
    and retrieving specific result data.
    """

    @staticmethod
    def get_best_match(result: "ActionResult") -> Optional["Match"]:
        """Find the match with the highest similarity score.

        Args:
            result: ActionResult to search

        Returns:
            Best scoring match, or None if no matches
        """
        if not result.match_list:
            return None
        return max(
            result.match_list,
            key=lambda m: m.get_score() if hasattr(m, "get_score") else 0,
        )

    @staticmethod
    def get_best_location(result: "ActionResult") -> Optional["Location"]:
        """Get the target location of the best scoring match.

        Args:
            result: ActionResult to extract from

        Returns:
            Location from best match, or None if no matches
        """
        from ..model.element.location import Location

        best = ResultExtractor.get_best_match(result)
        if best and hasattr(best, "get_target"):
            target = best.get_target()
            if isinstance(target, Location):
                return target
        return None

    @staticmethod
    def get_match_locations(result: "ActionResult") -> list["Location"]:
        """Get all match target locations.

        Args:
            result: ActionResult to extract from

        Returns:
            List of locations from all matches
        """
        locations = []
        for match in result.match_list:
            if hasattr(match, "get_target"):
                target = match.get_target()
                if target:
                    locations.append(target)
        return locations

    @staticmethod
    def size(result: "ActionResult") -> int:
        """Get the number of matches found.

        Args:
            result: ActionResult to count

        Returns:
            Count of matches in the result
        """
        return len(result.match_list)

    @staticmethod
    def is_empty(result: "ActionResult") -> bool:
        """Check if the action found any matches.

        Args:
            result: ActionResult to check

        Returns:
            True if no matches were found
        """
        return len(result.match_list) == 0

    @staticmethod
    def get_success_symbol(result: "ActionResult") -> str:
        """Get a visual symbol representing action success or failure.

        Args:
            result: ActionResult to evaluate

        Returns:
            Unicode symbol indicating success (✓) or failure (✗)
        """
        return "✓" if result.success else "✗"

    @staticmethod
    def get_summary(result: "ActionResult") -> str:
        """Get a summary of the action result.

        Args:
            result: ActionResult to summarize

        Returns:
            Multi-line summary string
        """
        summary = []
        if result.action_config:
            summary.append(f"Action: {result.action_config.__class__.__name__}")
        summary.append(f"Success: {result.success}")
        summary.append(f"Number of matches: {ResultExtractor.size(result)}")
        if result.active_states:
            summary.append(f"Active states: {', '.join(result.active_states)}")
        if result.text:
            summary.append(f"Extracted text: {result.text}")
        return "\n".join(summary)

    @staticmethod
    def print_matches(result: "ActionResult") -> None:
        """Print all matches to standard output.

        Args:
            result: ActionResult containing matches to print
        """
        for match in result.match_list:
            print(match)
