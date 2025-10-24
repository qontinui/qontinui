"""Find class - ported from Qontinui framework.

Base class for all find operations with builder pattern.
"""

from __future__ import annotations

import time
from typing import Any

from ..actions import FindOptions
from ..model.element import Image, Location, Pattern, Region
from ..model.search_regions import SearchRegions
from .find_results import FindResults
from .match import Match
from .matches import Matches

# Type alias for search regions
SearchRegion = Region | SearchRegions


class Find:
    """Base find class with builder pattern.

    Port of Find from Qontinui framework class.
    Provides fluent interface for configuring and executing find operations.
    """

    def __init__(self, target: Pattern | Image | str | None = None):
        """Initialize Find with optional target.

        Args:
            target: Pattern, Image, or image path to find
        """
        # Pattern now has mask support built-in
        self._target: Pattern | None
        if isinstance(target, Pattern):
            self._target = target
        else:
            self._target = self._convert_to_pattern(target) if target else None

        self._options = FindOptions()
        self._search_region: SearchRegion | None = None
        self._screenshot: Any | None = None
        self._method = "template"  # Default matching method

    def _convert_to_pattern(self, target: Pattern | Image | str) -> Pattern:
        """Convert various input types to Pattern.

        Args:
            target: Pattern, Image, or path string

        Returns:
            Pattern object with full mask
        """
        if isinstance(target, Pattern):
            return target
        elif isinstance(target, Image):
            return Pattern.from_image(target)
        elif isinstance(target, str):
            return Pattern.from_file(target)
        else:
            raise ValueError(f"Invalid target type: {type(target)}")

    def pattern(self, pattern: Pattern) -> Find:
        """Set the pattern to find (fluent).

        Args:
            pattern: Pattern to search for

        Returns:
            Self for chaining
        """
        self._target = pattern
        return self

    def image(self, image: Image) -> Find:
        """Set the image to find (fluent).

        Args:
            image: Image to search for

        Returns:
            Self for chaining
        """
        self._target = Pattern.from_image(image)
        return self

    def similarity(self, similarity: float) -> Find:
        """Set minimum similarity threshold (fluent).

        Args:
            similarity: Minimum similarity (0.0 to 1.0)

        Returns:
            Self for chaining
        """
        self._options.min_similarity(similarity)
        if self._target:
            # Pattern always has similarity_threshold attribute
            self._target.similarity_threshold = similarity
        return self

    def search_region(self, region: Region | SearchRegions) -> Find:
        """Set search region (fluent).

        Args:
            region: Region to search within

        Returns:
            Self for chaining
        """
        if isinstance(region, SearchRegions):
            self._search_region = region
            # SearchRegions doesn't have x, y, width, height directly
            # Don't set individual region in options
        elif isinstance(region, Region):
            self._search_region = Region(region.x, region.y, region.width, region.height)
            self._options.search_region(region.x, region.y, region.width, region.height)
        return self

    def timeout(self, seconds: float) -> Find:
        """Set timeout for find operation (fluent).

        Args:
            seconds: Timeout in seconds

        Returns:
            Self for chaining
        """
        self._options.timeout(seconds)
        return self

    def max_matches(self, count: int) -> Find:
        """Set maximum number of matches to find (fluent).

        Args:
            count: Maximum matches

        Returns:
            Self for chaining
        """
        self._options.max_matches(count)
        return self

    def find_all(self, find_all: bool = True) -> Find:
        """Enable/disable finding all matches (fluent).

        Args:
            find_all: True to find all matches

        Returns:
            Self for chaining
        """
        self._options.find_all(find_all)
        return self

    def method(self, method: str) -> Find:
        """Set matching method (fluent).

        Args:
            method: Matching method ('template', 'feature', 'ai')

        Returns:
            Self for chaining
        """
        self._method = method
        return self

    def screenshot(self, screenshot: Any) -> Find:
        """Set screenshot to search in (fluent).

        Args:
            screenshot: Screenshot image

        Returns:
            Self for chaining
        """
        self._screenshot = screenshot
        return self

    def sort_by(self, criteria: str) -> Find:
        """Set sort criteria for results (fluent).

        Args:
            criteria: Sort criteria ('similarity', 'position', 'size')

        Returns:
            Self for chaining
        """
        self._options.sort_by(criteria)
        return self

    def cache_result(self, cache: bool = True) -> Find:
        """Enable/disable result caching (fluent).

        Args:
            cache: True to cache results

        Returns:
            Self for chaining
        """
        self._options.cache_result(cache)
        return self

    def use_cache(self, use: bool = True) -> Find:
        """Enable/disable cache usage (fluent).

        Args:
            use: True to use cached results

        Returns:
            Self for chaining
        """
        self._options.use_cache(use)
        return self

    def execute(self) -> FindResults:
        """Execute the find operation.

        Returns:
            FindResults with matches
        """
        if not self._target:
            return FindResults.empty()

        start_time = time.time()

        # Get screenshot if not provided
        if self._screenshot is None:
            self._screenshot = self._capture_screenshot()

        # Perform the actual finding
        matches = self._perform_find()

        # Sort matches if configured
        sort_criteria = self._options._sort_by
        if sort_criteria == "similarity":
            matches.sort_by_similarity()
        elif sort_criteria == "position":
            matches.sort_by_position()

        # Apply max matches limit
        if not self._options._find_all and matches.size() > 0:
            first_match = matches.first
            if first_match is not None:
                matches = Matches([first_match])
        elif self._options._max_matches < matches.size():
            matches = Matches(matches.to_list()[: self._options._max_matches])

        duration = time.time() - start_time

        # Convert SearchRegions to Region if needed
        search_region_for_results = (
            self._search_region.regions[0]
            if isinstance(self._search_region, SearchRegions) and self._search_region.regions
            else self._search_region if isinstance(self._search_region, Region) else None
        )

        return FindResults(
            matches=matches,
            pattern=self._target,
            search_region=search_region_for_results,
            duration=duration,
            screenshot=self._screenshot,
            method=self._method,
        )

    def find(self) -> Match | None:
        """Find first/best match.

        Returns:
            First match or None
        """
        results = self.find_all(False).execute()
        return results.first_match

    def find_all_matches(self) -> Matches:
        """Find all matches.

        Returns:
            Matches collection
        """
        results = self.find_all(True).execute()
        return results.matches

    def exists(self) -> bool:
        """Check if pattern exists.

        Returns:
            True if pattern is found
        """
        match = self.find()
        return match is not None and match.exists()

    def wait_until_exists(self, timeout: float = 10.0) -> Match | None:
        """Wait until pattern appears.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Match when found or None if timeout
        """
        start_time = time.time()
        check_interval = 0.5

        while time.time() - start_time < timeout:
            match = self.find()
            if match and match.exists():
                return match
            time.sleep(check_interval)

        return None

    def wait_until_vanishes(self, timeout: float = 10.0) -> bool:
        """Wait until pattern disappears.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if pattern vanished, False if timeout
        """
        start_time = time.time()
        check_interval = 0.5

        while time.time() - start_time < timeout:
            if not self.exists():
                return True
            time.sleep(check_interval)

        return False

    def _capture_screenshot(self) -> Any:
        """Capture screenshot for finding.

        Returns:
            Screenshot image
        """
        # This would capture actual screenshot
        # For now, return placeholder
        from ..actions.pure import PureActions

        pure = PureActions()
        region_tuple = None
        if self._search_region is not None:
            if isinstance(self._search_region, Region):
                region_tuple = (
                    self._search_region.x,
                    self._search_region.y,
                    self._search_region.width,
                    self._search_region.height,
                )
            # SearchRegions doesn't have to_tuple method
        result = pure.screenshot(region=region_tuple)
        return result.data if result.success else None

    def _perform_find(self) -> Matches:
        """Perform the actual pattern matching.

        Returns:
            Matches found
        """
        if not self._target or not self._screenshot:
            return Matches()

        # Import required modules
        import cv2
        import numpy as np

        from ..model.match import Match as MatchObject

        # All Patterns now have pixel_data and mask
        template = self._target.pixel_data
        pattern_mask = self._target.mask

        if template is None:
            return Matches()

        # Get screenshot as numpy array
        if hasattr(self._screenshot, "get_mat_bgr"):
            screenshot = self._screenshot.get_mat_bgr()
        elif isinstance(self._screenshot, np.ndarray):
            screenshot = self._screenshot
        else:
            return Matches()

        if screenshot is None:
            return Matches()

        # Apply search region if specified
        search_img = screenshot
        offset_x, offset_y = 0, 0
        if self._search_region and isinstance(self._search_region, Region):
            x, y = self._search_region.x, self._search_region.y
            w, h = self._search_region.width, self._search_region.height
            # Ensure region is within bounds
            x = max(0, min(x, screenshot.shape[1]))
            y = max(0, min(y, screenshot.shape[0]))
            w = min(w, screenshot.shape[1] - x)
            h = min(h, screenshot.shape[0] - y)
            search_img = screenshot[y : y + h, x : x + w]
            offset_x, offset_y = x, y

        # Perform template matching
        if self._method == "template":
            # Store original template dimensions
            template_height, template_width = template.shape[:2]

            # Determine which mask to use (priority: pattern_mask > alpha channel)
            mask = None
            if pattern_mask is not None:
                # Use the pattern mask (from MaskedPattern)
                # Convert mask to uint8 format (0-255) if it's in float format (0.0-1.0)
                if pattern_mask.dtype == np.float32 or pattern_mask.dtype == np.float64:
                    mask = (pattern_mask * 255).astype(np.uint8)
                else:
                    mask = pattern_mask.astype(np.uint8)
            elif template.shape[2] == 4 if len(template.shape) == 3 else False:
                # Fall back to alpha channel if present and no pattern mask
                alpha = template[:, :, 3]
                mask = alpha.copy()

            # Prepare template (remove alpha channel if present)
            if len(template.shape) == 3 and template.shape[2] == 4:
                template_bgr = template[:, :, :3]
            else:
                template_bgr = template

            # Ensure screenshot is BGR (remove alpha if present)
            if len(search_img.shape) == 3 and search_img.shape[2] == 4:
                search_img = search_img[:, :, :3]

            # Perform template matching with or without mask
            if mask is not None:
                result = cv2.matchTemplate(
                    search_img, template_bgr, cv2.TM_CCOEFF_NORMED, mask=mask
                )
            else:
                result = cv2.matchTemplate(search_img, template_bgr, cv2.TM_CCOEFF_NORMED)

            matches_list = []
            min_similarity = self._options._min_similarity

            if self._options._find_all:
                # Find all matches above threshold
                locations = np.where(result >= min_similarity)

                # Collect all potential matches
                for pt in zip(*locations[::-1], strict=False):  # Switch x and y
                    score = result[pt[1], pt[0]]
                    x = int(pt[0]) + offset_x
                    y = int(pt[1]) + offset_y

                    # Debug: Check for inf scores
                    import math

                    if not math.isfinite(float(score)):
                        print(f"[Find] WARNING: Non-finite score detected at ({x}, {y}): {score}")
                        continue  # Skip non-finite scores

                    match_obj = MatchObject(
                        target=Location(region=Region(x, y, template_width, template_height)),
                        score=float(score),
                        name=self._target.name if self._target else "match",
                    )
                    matches_list.append(Match(match_obj))

                # Apply Non-Maximum Suppression to remove overlapping matches
                if len(matches_list) > 1:
                    matches_list = self._apply_nms(matches_list, overlap_threshold=0.3)

                # Final filter to ensure all matches meet minimum similarity
                print(f"[Find] Before final filter: {len(matches_list)} matches")
                for i, m in enumerate(matches_list):
                    print(
                        f"[Find]   Match {i}: similarity={m.similarity}, min_required={min_similarity}"
                    )

                matches_list = [m for m in matches_list if m.similarity >= min_similarity]
                print(f"[Find] After final filter: {len(matches_list)} matches")

                # Filter by search regions using precedence hierarchy
                # Precedence: 1. Options (self._search_region), 2. Pattern, 3. StateImage, 4. whole screen
                search_regions_to_use: SearchRegions | None = None

                # Level 1: Check Options-level search regions (highest priority)
                if isinstance(self._search_region, SearchRegions) and self._search_region.regions:
                    search_regions_to_use = self._search_region
                    print(
                        f"[Find] Using Options-level search regions ({len(self._search_region.regions)} regions)"
                    )

                # Level 2: Check Pattern-level search regions
                elif (
                    self._target
                    and hasattr(self._target, "search_regions")
                    and self._target.search_regions
                ):
                    # Pattern's search_regions is already a SearchRegions object with Region objects
                    if self._target.search_regions.regions:
                        search_regions_to_use = self._target.search_regions
                        print(
                            f"[Find] Using Pattern-level search regions ({len(self._target.search_regions.regions)} regions)"
                        )

                # Level 3: StateImage-level search regions would be passed via Options
                # (StateImage.find() should set Options with its search_regions)

                # Level 4: Whole screen (no filtering) - search_regions_to_use remains None

                # Apply search region filtering if we have regions to use
                if search_regions_to_use and search_regions_to_use.regions:
                    filtered_matches = []
                    for match in matches_list:
                        if match.region:
                            match_center = match.center
                            # Check if match center is in ANY of the search regions
                            for search_region in search_regions_to_use.regions:
                                if search_region.contains(match_center):
                                    filtered_matches.append(match)
                                    break  # Found in at least one region, move to next match
                    matches_list = filtered_matches
                    print(f"[Find] After search region filter: {len(matches_list)} matches")
            else:
                # Find best match only
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                print(f"[Find] BEST MATCH: confidence={max_val}, min_similarity={min_similarity}, passes={max_val >= min_similarity}")

                if max_val >= min_similarity:
                    x = int(max_loc[0]) + offset_x
                    y = int(max_loc[1]) + offset_y

                    match_obj = MatchObject(
                        target=Location(region=Region(x, y, template_width, template_height)),
                        score=float(max_val),
                        name=self._target.name if self._target else "match",
                    )
                    matches_list.append(Match(match_obj))
                    print(f"[Find] Added match at ({x}, {y}) with confidence {max_val}")
                else:
                    print(f"[Find] Match rejected: {max_val} < {min_similarity}")

            return Matches(matches_list)

        # Other methods not implemented yet
        return Matches()

    def _apply_nms(self, matches: list[Match], overlap_threshold: float = 0.3) -> list[Match]:
        """Apply Non-Maximum Suppression to remove overlapping matches.

        Args:
            matches: List of matches to filter
            overlap_threshold: IoU threshold for considering matches as overlapping

        Returns:
            Filtered list of matches
        """
        if not matches:
            return matches

        # Sort by score descending
        sorted_matches = sorted(matches, key=lambda m: m.similarity, reverse=True)

        kept_matches: list[Match] = []
        for match in sorted_matches:
            # Skip matches without regions
            if match.region is None:
                continue

            # Check if this match overlaps with any kept match
            should_keep = True
            for kept in kept_matches:
                # Skip kept matches without regions
                if kept.region is None:
                    continue

                # Calculate IoU (Intersection over Union)
                x1 = max(match.region.x, kept.region.x)
                y1 = max(match.region.y, kept.region.y)
                x2 = min(match.region.x + match.region.width, kept.region.x + kept.region.width)
                y2 = min(match.region.y + match.region.height, kept.region.y + kept.region.height)

                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = match.region.width * match.region.height
                    area2 = kept.region.width * kept.region.height
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0

                    if iou > overlap_threshold:
                        should_keep = False
                        break

            if should_keep:
                kept_matches.append(match)

        return kept_matches

    def __str__(self) -> str:
        """String representation."""
        target_name = self._target.name if self._target else "None"
        return f"Find(target='{target_name}', method='{self._method}')"

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"Find(target={self._target}, " f"method='{self._method}', " f"options={self._options})"
        )
