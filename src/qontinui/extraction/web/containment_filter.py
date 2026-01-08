"""
Containment Filter for web extraction.

Filters out container elements that fully encompass other elements
without having unique visual content. Keeps only "atomic" elements
that represent actual UI components.
"""

import logging
from dataclasses import dataclass
from typing import Any

from .models import BoundingBox, RawElement

logger = logging.getLogger(__name__)


@dataclass
class ContainmentInfo:
    """Information about an element's containment relationships."""

    element_id: str
    contained_element_ids: list[str]
    contained_count: int
    is_container: bool
    container_reason: str | None = None
    keep_reason: str | None = None


class ContainmentFilter:
    """
    Filters out container elements, keeping only atomic elements.

    An element is considered a "container" (to be filtered out) if:
    1. Its bbox fully encompasses 2+ other elements
    2. It has no unique visual content (no text, no background, no border)
    3. It's not interactive

    Elements are kept if they have:
    - Visible border or background
    - Direct text content (not just child text)
    - Interactive functionality (button, input, link)
    - Are leaf nodes (contain no other elements)
    """

    # Tags that are always containers (structural only)
    STRUCTURAL_TAGS = frozenset(
        {
            "div",
            "span",
            "section",
            "article",
            "main",
            "aside",
            "header",
            "footer",
            "nav",
            "figure",
            "figcaption",
        }
    )

    # Tags that should always be kept even if they contain elements
    ALWAYS_KEEP_TAGS = frozenset(
        {
            "button",
            "a",
            "input",
            "select",
            "textarea",
            "label",
            "img",
            "svg",
            "canvas",
            "video",
            "audio",
            "iframe",
            "table",
            "ul",
            "ol",
            "li",
            "dl",
            "dt",
            "dd",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "pre",
            "code",
        }
    )

    def __init__(
        self,
        containment_threshold: int = 2,
        min_overlap_ratio: float = 0.95,
        keep_visually_distinct: bool = True,
    ) -> None:
        """
        Initialize the containment filter.

        Args:
            containment_threshold: Minimum number of contained elements
                to consider an element a container. Default 2.
            min_overlap_ratio: Minimum overlap ratio to consider
                one element fully contained in another. Default 0.95.
            keep_visually_distinct: If True, keep elements that have
                visible styling even if they contain other elements.
        """
        self.containment_threshold = containment_threshold
        self.min_overlap_ratio = min_overlap_ratio
        self.keep_visually_distinct = keep_visually_distinct

    def filter_elements(
        self,
        elements: list[RawElement],
    ) -> tuple[list[RawElement], list[ContainmentInfo]]:
        """
        Filter out container elements.

        Args:
            elements: List of raw elements to filter.

        Returns:
            Tuple of (filtered_elements, containment_info).
            - filtered_elements: Elements that passed the filter.
            - containment_info: Analysis of each element's containment.
        """
        if not elements:
            return [], []

        logger.info(f"Filtering {len(elements)} elements for containment...")

        # Build containment relationships
        containment_map = self._build_containment_map(elements)

        # Analyze each element
        containment_info: list[ContainmentInfo] = []
        elements_by_id = {e.id: e for e in elements}

        for element in elements:
            contained_ids = containment_map.get(element.id, [])
            info = self._analyze_element(element, contained_ids, elements_by_id)
            containment_info.append(info)

            # Update the element's is_container flag
            element.is_container = info.is_container

        # Filter out containers
        filtered = [e for e in elements if not e.is_container]

        container_count = len(elements) - len(filtered)
        logger.info(
            f"Filtered {container_count} containers, " f"keeping {len(filtered)} atomic elements"
        )

        return filtered, containment_info

    def _build_containment_map(
        self,
        elements: list[RawElement],
    ) -> dict[str, list[str]]:
        """
        Build a map of element ID -> list of contained element IDs.

        Uses spatial containment based on bounding boxes.
        """
        containment: dict[str, list[str]] = {e.id: [] for e in elements}

        # Sort by area (largest first) for efficiency
        sorted_elements = sorted(
            elements,
            key=lambda e: e.bbox.width * e.bbox.height,
            reverse=True,
        )

        # Check each pair for containment
        for i, outer in enumerate(sorted_elements):
            for inner in sorted_elements[i + 1 :]:
                # Skip if same element or inner is larger (can't be contained)
                if inner.id == outer.id:
                    continue

                if self._bbox_contains(outer.bbox, inner.bbox):
                    containment[outer.id].append(inner.id)

        return containment

    def _bbox_contains(
        self,
        outer: BoundingBox,
        inner: BoundingBox,
    ) -> bool:
        """
        Check if outer bbox fully contains inner bbox.

        Uses overlap ratio to handle near-containment cases.
        """
        # Calculate intersection
        x1 = max(outer.x, inner.x)
        y1 = max(outer.y, inner.y)
        x2 = min(outer.x + outer.width, inner.x + inner.width)
        y2 = min(outer.y + outer.height, inner.y + inner.height)

        if x2 <= x1 or y2 <= y1:
            return False  # No intersection

        intersection_area = (x2 - x1) * (y2 - y1)
        inner_area = inner.width * inner.height

        if inner_area == 0:
            return False

        overlap_ratio = intersection_area / inner_area
        return overlap_ratio >= self.min_overlap_ratio

    def _analyze_element(
        self,
        element: RawElement,
        contained_ids: list[str],
        elements_by_id: dict[str, RawElement],
    ) -> ContainmentInfo:
        """
        Analyze an element to determine if it's a container.

        Returns ContainmentInfo with the analysis results.
        """
        contained_count = len(contained_ids)

        # Rule 1: Always keep certain tags
        if element.tag_name in self.ALWAYS_KEEP_TAGS:
            return ContainmentInfo(
                element_id=element.id,
                contained_element_ids=contained_ids,
                contained_count=contained_count,
                is_container=False,
                keep_reason=f"tag:{element.tag_name}",
            )

        # Rule 2: Leaf elements are never containers
        if contained_count == 0:
            return ContainmentInfo(
                element_id=element.id,
                contained_element_ids=[],
                contained_count=0,
                is_container=False,
                keep_reason="leaf_element",
            )

        # Rule 3: Below containment threshold
        if contained_count < self.containment_threshold:
            return ContainmentInfo(
                element_id=element.id,
                contained_element_ids=contained_ids,
                contained_count=contained_count,
                is_container=False,
                keep_reason=f"below_threshold:{contained_count}<{self.containment_threshold}",
            )

        # Rule 4: Interactive elements are kept
        if element.is_interactive:
            return ContainmentInfo(
                element_id=element.id,
                contained_element_ids=contained_ids,
                contained_count=contained_count,
                is_container=False,
                keep_reason="interactive",
            )

        # Rule 5: Elements with direct text content are kept
        if element.text_content and len(element.text_content.strip()) > 0:
            return ContainmentInfo(
                element_id=element.id,
                contained_element_ids=contained_ids,
                contained_count=contained_count,
                is_container=False,
                keep_reason="has_direct_text",
            )

        # Rule 6: Keep visually distinct elements
        if self.keep_visually_distinct and self._has_visual_distinction(element):
            return ContainmentInfo(
                element_id=element.id,
                contained_element_ids=contained_ids,
                contained_count=contained_count,
                is_container=False,
                keep_reason="visually_distinct",
            )

        # Rule 7: Structural tags without visual content are containers
        if element.tag_name in self.STRUCTURAL_TAGS:
            return ContainmentInfo(
                element_id=element.id,
                contained_element_ids=contained_ids,
                contained_count=contained_count,
                is_container=True,
                container_reason=f"structural_tag:{element.tag_name}",
            )

        # Default: Mark as container if it contains multiple elements
        # and has no distinguishing features
        return ContainmentInfo(
            element_id=element.id,
            contained_element_ids=contained_ids,
            contained_count=contained_count,
            is_container=True,
            container_reason=f"contains:{contained_count}_elements",
        )

    def _has_visual_distinction(self, element: RawElement) -> bool:
        """
        Check if element has visual properties that make it distinct.

        Returns True if element has:
        - Non-transparent background
        - Visible border
        - Box shadow
        - Specific styling
        """
        # Check background color
        if element.background_color:
            r, g, b, a = element.background_color
            # Non-transparent and not pure white
            if a > 10 and not (r > 250 and g > 250 and b > 250):
                return True

        # Check border color (indicates visible border)
        if element.border_color:
            _, _, _, a = element.border_color
            if a > 10:
                return True

        # Check computed styles for visual distinction
        styles = element.computed_styles
        if styles:
            # Has box shadow
            if styles.get("boxShadow") and styles["boxShadow"] != "none":
                return True

            # Has border radius (indicates styled component)
            border_radius = styles.get("borderRadius", "0")
            if border_radius and border_radius != "0px":
                return True

            # Has specific positioning (could be a UI component)
            position = styles.get("position", "static")
            if position in ("fixed", "sticky"):
                return True

        return False

    def get_containment_hierarchy(
        self,
        elements: list[RawElement],
    ) -> dict[str, dict[str, Any]]:
        """
        Build a containment hierarchy for debugging/visualization.

        Returns a dict mapping element ID to:
        - contained_by: list of parent element IDs
        - contains: list of child element IDs
        - depth: containment depth (0 = root)
        """
        containment_map = self._build_containment_map(elements)

        # Build reverse map (contained_by)
        contained_by: dict[str, list[str]] = {e.id: [] for e in elements}
        for outer_id, inner_ids in containment_map.items():
            for inner_id in inner_ids:
                contained_by[inner_id].append(outer_id)

        # Calculate depths
        depths: dict[str, int] = {}

        def calc_depth(element_id: str, visited: set[str]) -> int:
            if element_id in depths:
                return depths[element_id]
            if element_id in visited:
                return 0  # Cycle detection

            visited.add(element_id)
            parents = contained_by.get(element_id, [])

            if not parents:
                depths[element_id] = 0
            else:
                depths[element_id] = 1 + max(calc_depth(p, visited) for p in parents)

            return depths[element_id]

        for element in elements:
            calc_depth(element.id, set())

        # Build result
        result = {}
        for element in elements:
            result[element.id] = {
                "tag": element.tag_name,
                "contained_by": contained_by.get(element.id, []),
                "contains": containment_map.get(element.id, []),
                "depth": depths.get(element.id, 0),
                "is_container": element.is_container,
            }

        return result
