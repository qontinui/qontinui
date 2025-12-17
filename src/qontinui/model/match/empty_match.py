"""EmptyMatch class - ported from Qontinui framework.

Represents the absence of a match in the framework.
"""

from ..element import Image, Region, Scene
from .match import Match as MatchObject


class EmptyMatch:
    """Represents the absence of a match.

    Port of EmptyMatch from Qontinui framework class.

    EmptyMatch is a specialized Match that explicitly represents failed search operations.
    Rather than using null values or empty results, EmptyMatch provides a concrete object that maintains
    the Match interface while clearly indicating that no visual element was found. This approach
    follows the Null Object pattern, enabling cleaner code without null checks.

    Key characteristics:
    - Zero Score: Always has a match score of 0 indicating no similarity
    - Empty Region: Contains a 0x0 region at position (0,0)
    - Named Identity: Carries "no match" as its identifying name
    - Valid Object: Can be used in all contexts expecting a Match

    Benefits of EmptyMatch pattern:
    - Eliminates null pointer exceptions in match processing
    - Enables uniform handling of successful and failed searches
    - Provides context about what was searched even when nothing was found
    - Simplifies conditional logic in automation scripts

    Use cases:
    - Representing failed Find operations without using null
    - Placeholder in collections when some searches fail
    - Default values in match-based data structures
    - Testing and mock scenarios requiring explicit non-matches
    """

    def __init__(
        self,
        name: str = "no match",
        search_image: Image | None = None,
        scene: Scene | None = None,
    ) -> None:
        """Initialize EmptyMatch.

        Args:
            name: Name for the empty match
            search_image: Image that was searched for
            scene: Scene that was searched
        """
        self.name = name
        self.region = Region(0, 0, 0, 0)
        self.score = 0.0
        self.search_image = search_image
        self.scene = scene
        self.state_object_data = None

    def to_match_object(self) -> MatchObject:
        """Convert to MatchObject.

        Returns:
            MatchObject with empty values
        """
        from ..element import Location

        return MatchObject(
            target=Location(0, 0),
            score=self.score,
            search_image=self.search_image,
            name=self.name,
        )

    def is_empty(self) -> bool:
        """Check if this is an empty match.

        Returns:
            Always True for EmptyMatch
        """
        return True

    def exists(self) -> bool:
        """Check if match exists.

        Returns:
            Always False for EmptyMatch
        """
        return False

    def __bool__(self) -> bool:
        """Boolean evaluation.

        Returns:
            Always False for EmptyMatch
        """
        return False

    def __str__(self) -> str:
        """String representation."""
        return f"EmptyMatch('{self.name}')"

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"EmptyMatch(name='{self.name}', region={self.region}, score={self.score})"
        )

    @classmethod
    def builder(cls) -> "EmptyMatchBuilder":
        """Create a new EmptyMatchBuilder.

        Returns:
            New builder instance
        """
        return EmptyMatchBuilder()


class EmptyMatchBuilder:
    """Builder for EmptyMatch class."""

    def __init__(self) -> None:
        """Initialize builder with defaults."""
        self.name = "no match"
        self.region = Region(0, 0, 0, 0)
        self.search_image = None
        self.scene = None
        self.state_object_data = None

    def set_name(self, name: str) -> "EmptyMatchBuilder":
        """Set match name (fluent).

        Args:
            name: Name for the empty match

        Returns:
            Self for chaining
        """
        self.name = name
        return self

    def set_region(self, region: Region) -> "EmptyMatchBuilder":
        """Set region (fluent).

        Args:
            region: Region for the empty match

        Returns:
            Self for chaining
        """
        self.region = region
        return self

    def set_search_image(self, image: Image) -> "EmptyMatchBuilder":
        """Set search image (fluent).

        Args:
            image: Image that was searched for

        Returns:
            Self for chaining
        """
        self.search_image = image  # type: ignore[assignment]
        return self

    def set_scene(self, scene: Scene) -> "EmptyMatchBuilder":
        """Set scene (fluent).

        Args:
            scene: Scene that was searched

        Returns:
            Self for chaining
        """
        self.scene = scene  # type: ignore[assignment]
        return self

    def build(self) -> EmptyMatch:
        """Build the EmptyMatch.

        Returns:
            Configured EmptyMatch instance
        """
        match = EmptyMatch(
            name=self.name, search_image=self.search_image, scene=self.scene
        )
        match.region = self.region
        return match
