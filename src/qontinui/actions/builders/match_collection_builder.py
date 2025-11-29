"""Match collection builder for ObjectCollection.

Handles Matches and ActionResults.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...model.match import Match
    from ...model.state import StateImage, StateRegion
    from ..action_result import ActionResult


class MatchCollectionBuilder:
    """Builder for match-related objects in ObjectCollection.

    Handles:
    - ActionResults (matches)
    - Match objects converted to regions
    - Match objects converted to state images
    """

    def __init__(self) -> None:
        """Initialize builder with empty lists."""
        self.matches: list[ActionResult] = []
        self.state_regions_from_matches: list[StateRegion] = []
        self.state_images_from_matches: list[StateImage] = []

    def with_matches(self, *matches: "ActionResult") -> "MatchCollectionBuilder":
        """Add matches to collection.

        Args:
            matches: Variable number of ActionResult objects

        Returns:
            This builder for method chaining
        """
        self.matches.extend(matches)
        return self

    def set_matches(self, matches: list["ActionResult"]) -> "MatchCollectionBuilder":
        """Set matches list.

        Args:
            matches: List of ActionResult objects

        Returns:
            This builder for method chaining
        """
        self.matches = matches
        return self

    def with_match_objects_as_regions(self, *matches: "Match") -> "MatchCollectionBuilder":
        """Add match objects as regions.

        Args:
            matches: Variable number of Match objects

        Returns:
            This builder for method chaining
        """
        from ...model.state import StateRegion

        for match in matches:
            match_region = match.get_region()
            if match_region:
                state_region = StateRegion(region=match_region, name="match_region")
                self.state_regions_from_matches.append(state_region)
        return self

    def with_match_objects_as_state_images(self, *matches: "Match") -> "MatchCollectionBuilder":
        """Add match objects as state images.

        Args:
            matches: Variable number of Match objects

        Returns:
            This builder for method chaining
        """
        for match in matches:
            self.state_images_from_matches.append(match.to_state_image())  # type: ignore[arg-type]
        return self

    def build(self) -> tuple[list["ActionResult"], list["StateRegion"], list["StateImage"]]:
        """Build and return the matches and derived objects.

        Returns:
            Tuple of (matches, state_regions_from_matches, state_images_from_matches) copies
        """
        return (
            self.matches.copy(),
            self.state_regions_from_matches.copy(),
            self.state_images_from_matches.copy(),
        )
