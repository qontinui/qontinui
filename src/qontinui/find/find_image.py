"""FindImage class - ported from Qontinui framework.

Specialized find class for image-based pattern matching.

MIGRATION NOTE: This class now uses FindOptions image variant options
to delegate image processing to the new FindAction system.
"""

from ..actions.find import FindOptions as NewFindOptions
from ..model.element import Image, Location, Region
from .find import Find
from .find_results import FindResults
from .match import Match
from .matches import Matches


class FindImage(Find):
    """Specialized find class for image matching.

    MIGRATION: This class now uses FindOptions image variant options
    to configure image processing in the new FindAction system.
    """

    def __init__(self, image: Image | str | None = None) -> None:
        """Initialize FindImage with optional image.

        Args:
            image: Image or path to find
        """
        super().__init__()
        if image:
            if isinstance(image, str):
                self.image(Image.from_file(image))
            else:
                self.image(image)

        # Image-specific settings (now passed to FindOptions)
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
        self.search_region(region)
        return self

    def near(self, location: Location, radius: int) -> "FindImage":
        """Search near a specific location (fluent).

        Args:
            location: Center location
            radius: Search radius in pixels

        Returns:
            Self for chaining
        """
        region = Region(location.x - radius, location.y - radius, radius * 2, radius * 2)
        self.search_region(region)
        return self

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
            if ref_region is None:
                raise ValueError("Match does not have a region")
        else:
            ref_region = reference

        search_region = ref_region.above(distance)
        self.search_region(search_region)
        return self

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
            if ref_region is None:
                raise ValueError("Match does not have a region")
        else:
            ref_region = reference

        search_region = ref_region.below(distance)
        self.search_region(search_region)
        return self

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
            if ref_region is None:
                raise ValueError("Match does not have a region")
        else:
            ref_region = reference

        search_region = ref_region.left_of(distance)
        self.search_region(search_region)
        return self

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
            if ref_region is None:
                raise ValueError("Match does not have a region")
        else:
            ref_region = reference

        search_region = ref_region.right_of(distance)
        self.search_region(search_region)
        return self

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

    def execute(self) -> FindResults:
        """Execute the find operation with image variant options.

        MIGRATION: This method overrides Find.execute() to add image
        variant options (grayscale, edge detection, etc.) to FindOptions.

        Returns:
            FindResults with matches
        """
        from ..model.search_regions import SearchRegions

        if not self._target:
            return FindResults.empty()

        import time

        start_time = time.time()

        # Build FindOptions with image variant options
        search_region = None
        if isinstance(self._search_region, Region):
            search_region = self._search_region
        elif isinstance(self._search_region, SearchRegions) and self._search_region.regions:
            search_region = self._search_region.regions[0]

        options = NewFindOptions(
            similarity=self._min_similarity,
            find_all=self._find_all_mode,
            search_region=search_region,
            timeout=self._timeout,
            collect_debug=True,
            # Image variant options
            grayscale=self._use_grayscale,
            edge_detection=self._use_edges,
            scale_invariant=self._scale_invariant,
            rotation_invariant=self._rotation_invariant,
            color_tolerance=self._color_tolerance,
        )

        # Delegate to FindAction
        result = self._find_action.find(self._target, options)

        # Convert new Match objects to old Match wrapper objects
        old_matches = self._convert_matches(result.matches.to_list())

        # Create Matches collection
        matches = Matches(old_matches)

        # Sort matches if configured
        if self._sort_by == "similarity":
            matches.sort_by_similarity()
        elif self._sort_by == "position":
            matches.sort_by_position()

        # Apply max matches limit
        if not self._find_all_mode and matches.size() > 0:
            first_match = matches.first
            if first_match is not None:
                matches = Matches([first_match])
        elif self._max_matches < matches.size():
            matches = Matches(matches.to_list()[: self._max_matches])

        duration = time.time() - start_time

        # Convert SearchRegions to Region if needed for results
        search_region_for_results = (
            self._search_region.regions[0]
            if isinstance(self._search_region, SearchRegions) and self._search_region.regions
            else (self._search_region if isinstance(self._search_region, Region) else None)
        )

        return FindResults(
            matches=matches,
            pattern=self._target,
            search_region=search_region_for_results,
            duration=duration,
            screenshot=None,
            method=self._method,
        )

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
