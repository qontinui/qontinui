"""FindImage class - ported from Qontinui framework.

Specialized find class for image-based pattern matching.
"""

from ..model.element import Image, Location, Region
from .find import Find
from .match import Match
from .matches import Matches


class FindImage(Find):
    """Specialized find class for image matching.

    Port of FindImage from Qontinui framework class.
    Extends Find with image-specific matching capabilities.
    """

    def __init__(self, image: Image | str | None = None):
        """Initialize FindImage with optional image.

        Args:
            image: Image or path to find
        """
        super().__init__()
        if image:
            if isinstance(image, str):
                self.image(Image(name=image, path=image))
            else:
                self.image(image)

        # Image-specific settings
        self._use_grayscale = False
        self._use_edges = False
        self._scale_invariant = False
        self._rotation_invariant = False
        self._color_tolerance = 0

    def grayscale(self, use: bool = True) -> "FindImage":
        """Enable/disable grayscale matching (fluent).

        Args:
            use: True to use grayscale matching

        Returns:
            Self for chaining
        """
        self._use_grayscale = use
        return self

    def edges(self, use: bool = True) -> "FindImage":
        """Enable/disable edge-based matching (fluent).

        Args:
            use: True to use edge detection

        Returns:
            Self for chaining
        """
        self._use_edges = use
        return self

    def scale_invariant(self, enable: bool = True) -> "FindImage":
        """Enable/disable scale-invariant matching (fluent).

        Args:
            enable: True for scale-invariant matching

        Returns:
            Self for chaining
        """
        self._scale_invariant = enable
        return self

    def rotation_invariant(self, enable: bool = True) -> "FindImage":
        """Enable/disable rotation-invariant matching (fluent).

        Args:
            enable: True for rotation-invariant matching

        Returns:
            Self for chaining
        """
        self._rotation_invariant = enable
        return self

    def color_tolerance(self, tolerance: int) -> "FindImage":
        """Set color tolerance for matching (fluent).

        Args:
            tolerance: Color tolerance value (0-255)

        Returns:
            Self for chaining
        """
        self._color_tolerance = max(0, min(255, tolerance))
        return self

    def in_region(self, region: Region) -> "FindImage":
        """Set search region (fluent).

        Args:
            region: Region to search in

        Returns:
            Self for chaining
        """
        return self.search_region(region)

    def near(self, location: Location, radius: int) -> "FindImage":
        """Search near a specific location (fluent).

        Args:
            location: Center location
            radius: Search radius in pixels

        Returns:
            Self for chaining
        """
        region = Region(location.x - radius, location.y - radius, radius * 2, radius * 2)
        return self.search_region(region)

    def above(self, reference: Match | Region, distance: int = 100) -> "FindImage":
        """Search above a reference (fluent).

        Args:
            reference: Reference match or region
            distance: How far above to search

        Returns:
            Self for chaining
        """
        if isinstance(reference, Match):
            ref_region = reference.region
        else:
            ref_region = reference

        search_region = ref_region.above(distance)
        return self.search_region(search_region)

    def below(self, reference: Match | Region, distance: int = 100) -> "FindImage":
        """Search below a reference (fluent).

        Args:
            reference: Reference match or region
            distance: How far below to search

        Returns:
            Self for chaining
        """
        if isinstance(reference, Match):
            ref_region = reference.region
        else:
            ref_region = reference

        search_region = ref_region.below(distance)
        return self.search_region(search_region)

    def left_of(self, reference: Match | Region, distance: int = 100) -> "FindImage":
        """Search to the left of a reference (fluent).

        Args:
            reference: Reference match or region
            distance: How far left to search

        Returns:
            Self for chaining
        """
        if isinstance(reference, Match):
            ref_region = reference.region
        else:
            ref_region = reference

        search_region = ref_region.left_of(distance)
        return self.search_region(search_region)

    def right_of(self, reference: Match | Region, distance: int = 100) -> "FindImage":
        """Search to the right of a reference (fluent).

        Args:
            reference: Reference match or region
            distance: How far right to search

        Returns:
            Self for chaining
        """
        if isinstance(reference, Match):
            ref_region = reference.region
        else:
            ref_region = reference

        search_region = ref_region.right_of(distance)
        return self.search_region(search_region)

    def best_match(self) -> Match | None:
        """Find the best match by similarity.

        Returns:
            Best match or None
        """
        results = self.execute()
        return results.best_match

    def all_matches(self) -> list[Match]:
        """Find all matches as a list.

        Returns:
            List of all matches
        """
        results = self.find_all(True).execute()
        return results.matches.to_list()

    def count_matches(self) -> int:
        """Count number of matches.

        Returns:
            Number of matches found
        """
        results = self.find_all(True).execute()
        return results.count

    def _perform_find(self) -> Matches:
        """Perform image-specific pattern matching.

        Returns:
            Matches found
        """
        # Apply image-specific preprocessing
        if self._use_grayscale:
            # Convert to grayscale for matching
            pass

        if self._use_edges:
            # Apply edge detection
            pass

        if self._scale_invariant:
            # Use scale-invariant matching (e.g., SIFT/SURF)
            self._method = "feature"

        if self._rotation_invariant:
            # Use rotation-invariant matching
            self._method = "feature"

        # Call parent implementation
        return super()._perform_find()

    def highlight_matches(self, duration: float = 2.0) -> "FindImage":
        """Highlight all found matches on screen.

        Args:
            duration: How long to show highlights

        Returns:
            Self for chaining
        """
        results = self.execute()
        for match in results.matches:
            match.highlight(duration)
        return self

    def click_best(self) -> bool:
        """Click on the best match.

        Returns:
            True if clicked successfully
        """
        match = self.best_match()
        if match:
            result = match.click()
            return result.success
        return False

    def click_all(self) -> int:
        """Click on all matches.

        Returns:
            Number of successful clicks
        """
        matches = self.all_matches()
        successful = 0
        for match in matches:
            if match.click().success:
                successful += 1
        return successful

    def wait_and_click(self, timeout: float = 10.0) -> bool:
        """Wait for image to appear and click it.

        Args:
            timeout: Maximum wait time

        Returns:
            True if found and clicked
        """
        match = self.wait_until_exists(timeout)
        if match:
            result = match.click()
            return result.success
        return False

    def __str__(self) -> str:
        """String representation."""
        target_name = self._target.name if self._target else "None"
        return f"FindImage('{target_name}')"
