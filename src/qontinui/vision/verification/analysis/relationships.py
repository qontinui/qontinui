"""Element relationship analysis module.

Provides analysis of relationships between UI elements:
- Containment (parent/child)
- Proximity (nearby elements)
- Grouping (visually related elements)
- Associations (label-input pairs, etc.)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from qontinui_schemas.testing.assertions import BoundingBox

if TYPE_CHECKING:
    from qontinui_schemas.testing.environment import GUIEnvironment

    from qontinui.vision.verification.config import VisionConfig
    from qontinui.vision.verification.detection.ocr import OCRResult
    from qontinui.vision.verification.detection.template import TemplateMatch

logger = logging.getLogger(__name__)


class RelationshipType(str, Enum):
    """Types of element relationships."""

    CONTAINS = "contains"  # Element A contains element B
    CONTAINED_BY = "contained_by"  # Element A is inside element B
    SIBLING = "sibling"  # Same level, same parent
    ADJACENT = "adjacent"  # Directly next to each other
    NEARBY = "nearby"  # Within proximity threshold
    ALIGNED = "aligned"  # Share alignment axis
    LABELED_BY = "labeled_by"  # Text labels an element
    LABELS = "labels"  # Element is a label for another
    GROUP_MEMBER = "group_member"  # Part of a visual group


@dataclass
class ElementRelationship:
    """Represents a relationship between two elements."""

    source_id: str
    target_id: str
    relationship_type: RelationshipType
    confidence: float
    distance: float | None = None  # Pixel distance if applicable
    alignment_axis: str | None = None  # 'horizontal', 'vertical', 'both'
    metadata: dict = field(default_factory=dict)

    @property
    def is_spatial(self) -> bool:
        """Check if this is a spatial relationship."""
        return self.relationship_type in {
            RelationshipType.ADJACENT,
            RelationshipType.NEARBY,
            RelationshipType.ALIGNED,
        }

    @property
    def is_hierarchical(self) -> bool:
        """Check if this is a hierarchical relationship."""
        return self.relationship_type in {
            RelationshipType.CONTAINS,
            RelationshipType.CONTAINED_BY,
        }


@dataclass
class Element:
    """Represents a UI element for relationship analysis."""

    id: str
    bounds: BoundingBox
    element_type: str  # 'text', 'image', 'button', 'input', etc.
    text: str | None = None
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        """Get element center point."""
        return (
            self.bounds.x + self.bounds.width / 2,
            self.bounds.y + self.bounds.height / 2,
        )

    @property
    def area(self) -> int:
        """Get element area."""
        return int(self.bounds.width * self.bounds.height)


@dataclass
class ElementGroup:
    """A group of related elements."""

    id: str
    elements: list[Element]
    group_type: str  # 'form_field', 'menu', 'toolbar', 'card', 'list', etc.
    bounds: BoundingBox
    label: str | None = None
    confidence: float = 1.0
    relationships: list[ElementRelationship] = field(default_factory=list)

    @property
    def element_count(self) -> int:
        """Get number of elements in group."""
        return len(self.elements)

    @property
    def element_ids(self) -> list[str]:
        """Get IDs of all elements in group."""
        return [e.id for e in self.elements]


class RelationshipAnalyzer:
    """Analyzes relationships between UI elements.

    Detects various types of relationships:
    - Containment hierarchies (parent/child)
    - Spatial relationships (proximity, alignment)
    - Semantic relationships (label-input pairs)
    - Visual groupings (cards, lists, menus)

    Usage:
        analyzer = RelationshipAnalyzer(config, environment)

        # Analyze all relationships
        relationships = analyzer.find_all_relationships(elements)

        # Find specific relationships
        containers = analyzer.find_containers(element, elements)
        nearby = analyzer.find_nearby_elements(element, elements, threshold=50)

        # Group detection
        groups = analyzer.find_groups(elements)
    """

    def __init__(
        self,
        config: "VisionConfig | None" = None,
        environment: "GUIEnvironment | None" = None,
    ) -> None:
        """Initialize relationship analyzer.

        Args:
            config: Vision configuration.
            environment: GUI environment with component hints.
        """
        self._config = config
        self._environment = environment

        # Default thresholds
        self._containment_threshold = 0.9  # % of element must be inside
        self._proximity_threshold = 50  # pixels
        self._alignment_tolerance = 5  # pixels
        self._label_distance_threshold = 100  # max distance for label association

    def find_all_relationships(
        self,
        elements: list[Element],
    ) -> list[ElementRelationship]:
        """Find all relationships between elements.

        Args:
            elements: List of elements to analyze.

        Returns:
            List of all detected relationships.
        """
        relationships: list[ElementRelationship] = []

        for i, source in enumerate(elements):
            for j, target in enumerate(elements):
                if i == j:
                    continue

                # Check containment
                if self._contains(source.bounds, target.bounds):
                    relationships.append(
                        ElementRelationship(
                            source_id=source.id,
                            target_id=target.id,
                            relationship_type=RelationshipType.CONTAINS,
                            confidence=self._containment_confidence(source.bounds, target.bounds),
                        )
                    )

                # Check adjacency and proximity
                distance = self._edge_distance(source.bounds, target.bounds)
                if distance is not None:
                    if distance <= 5:
                        relationships.append(
                            ElementRelationship(
                                source_id=source.id,
                                target_id=target.id,
                                relationship_type=RelationshipType.ADJACENT,
                                confidence=1.0,
                                distance=distance,
                            )
                        )
                    elif distance <= self._proximity_threshold:
                        relationships.append(
                            ElementRelationship(
                                source_id=source.id,
                                target_id=target.id,
                                relationship_type=RelationshipType.NEARBY,
                                confidence=1.0 - (distance / self._proximity_threshold),
                                distance=distance,
                            )
                        )

                # Check alignment
                alignment = self._check_alignment(source.bounds, target.bounds)
                if alignment:
                    relationships.append(
                        ElementRelationship(
                            source_id=source.id,
                            target_id=target.id,
                            relationship_type=RelationshipType.ALIGNED,
                            confidence=1.0,
                            alignment_axis=alignment,
                        )
                    )

                # Check label relationships (text -> non-text)
                if source.element_type == "text" and target.element_type != "text":
                    if self._is_label_for(source, target):
                        relationships.append(
                            ElementRelationship(
                                source_id=source.id,
                                target_id=target.id,
                                relationship_type=RelationshipType.LABELS,
                                confidence=0.8,
                                distance=self._center_distance(source.bounds, target.bounds),
                            )
                        )

        return relationships

    def find_containers(
        self,
        element: Element,
        candidates: list[Element],
    ) -> list[Element]:
        """Find elements that contain the given element.

        Args:
            element: Element to find containers for.
            candidates: Candidate container elements.

        Returns:
            List of containing elements, sorted by area (smallest first).
        """
        containers = []
        for candidate in candidates:
            if candidate.id == element.id:
                continue
            if self._contains(candidate.bounds, element.bounds):
                containers.append(candidate)

        # Sort by area (smallest container first)
        containers.sort(key=lambda e: e.area)
        return containers

    def find_contained_elements(
        self,
        container: Element,
        candidates: list[Element],
    ) -> list[Element]:
        """Find elements contained within the given element.

        Args:
            container: Container element.
            candidates: Candidate elements to check.

        Returns:
            List of contained elements.
        """
        contained = []
        for candidate in candidates:
            if candidate.id == container.id:
                continue
            if self._contains(container.bounds, candidate.bounds):
                contained.append(candidate)

        return contained

    def find_nearby_elements(
        self,
        element: Element,
        candidates: list[Element],
        threshold: int | None = None,
    ) -> list[tuple[Element, float]]:
        """Find elements within proximity threshold.

        Args:
            element: Reference element.
            candidates: Candidate elements.
            threshold: Maximum distance in pixels.

        Returns:
            List of (element, distance) tuples, sorted by distance.
        """
        if threshold is None:
            threshold = self._proximity_threshold

        nearby: list[tuple[Element, float]] = []
        for candidate in candidates:
            if candidate.id == element.id:
                continue

            distance = self._edge_distance(element.bounds, candidate.bounds)
            if distance is not None and distance <= threshold:
                nearby.append((candidate, distance))

        # Sort by distance
        nearby.sort(key=lambda x: x[1])
        return nearby

    def find_aligned_elements(
        self,
        element: Element,
        candidates: list[Element],
        axis: str = "both",
    ) -> list[Element]:
        """Find elements aligned with the given element.

        Args:
            element: Reference element.
            candidates: Candidate elements.
            axis: Alignment axis ('horizontal', 'vertical', 'both').

        Returns:
            List of aligned elements.
        """
        aligned = []
        for candidate in candidates:
            if candidate.id == element.id:
                continue

            alignment = self._check_alignment(element.bounds, candidate.bounds)
            if alignment:
                if axis == "both" or axis == alignment or alignment == "both":
                    aligned.append(candidate)

        return aligned

    def find_label_for(
        self,
        element: Element,
        text_elements: list[Element],
    ) -> Element | None:
        """Find the label text for an element.

        Args:
            element: Element to find label for.
            text_elements: Candidate text elements.

        Returns:
            Label element if found.
        """
        best_label: Element | None = None
        best_score = 0.0

        for text_elem in text_elements:
            if text_elem.element_type != "text":
                continue

            if self._is_label_for(text_elem, element):
                score = self._label_score(text_elem, element)
                if score > best_score:
                    best_score = score
                    best_label = text_elem

        return best_label

    def find_groups(
        self,
        elements: list[Element],
    ) -> list[ElementGroup]:
        """Detect visual groups of related elements.

        Args:
            elements: Elements to group.

        Returns:
            List of detected element groups.
        """
        groups: list[ElementGroup] = []

        # Find form field groups (label + input)
        form_groups = self._find_form_field_groups(elements)
        groups.extend(form_groups)

        # Find list groups (vertically stacked similar items)
        list_groups = self._find_list_groups(elements)
        groups.extend(list_groups)

        # Find row groups (horizontally aligned items)
        row_groups = self._find_row_groups(elements)
        groups.extend(row_groups)

        return groups

    def find_siblings(
        self,
        element: Element,
        elements: list[Element],
    ) -> list[Element]:
        """Find sibling elements (same container, same level).

        Args:
            element: Reference element.
            elements: All elements to check.

        Returns:
            List of sibling elements.
        """
        # Find immediate container
        containers = self.find_containers(element, elements)
        if not containers:
            return []

        immediate_container = containers[0]

        # Find other elements in same container
        siblings = []
        for candidate in elements:
            if candidate.id == element.id:
                continue

            # Check if candidate is in the same container
            candidate_containers = self.find_containers(candidate, elements)
            if candidate_containers and candidate_containers[0].id == immediate_container.id:
                siblings.append(candidate)

        return siblings

    def _contains(self, outer: BoundingBox, inner: BoundingBox) -> bool:
        """Check if outer bounds contain inner bounds."""
        inner_left = inner.x
        inner_right = inner.x + inner.width
        inner_top = inner.y
        inner_bottom = inner.y + inner.height

        outer_left = outer.x
        outer_right = outer.x + outer.width
        outer_top = outer.y
        outer_bottom = outer.y + outer.height

        # Calculate overlap
        overlap_left = max(inner_left, outer_left)
        overlap_right = min(inner_right, outer_right)
        overlap_top = max(inner_top, outer_top)
        overlap_bottom = min(inner_bottom, outer_bottom)

        if overlap_right <= overlap_left or overlap_bottom <= overlap_top:
            return False

        overlap_area = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
        inner_area = inner.width * inner.height

        if inner_area == 0:
            return False

        containment_ratio = overlap_area / inner_area
        return bool(containment_ratio >= self._containment_threshold)

    def _containment_confidence(self, outer: BoundingBox, inner: BoundingBox) -> float:
        """Calculate containment confidence."""
        inner_left = inner.x
        inner_right = inner.x + inner.width
        inner_top = inner.y
        inner_bottom = inner.y + inner.height

        outer_left = outer.x
        outer_right = outer.x + outer.width
        outer_top = outer.y
        outer_bottom = outer.y + outer.height

        overlap_left = max(inner_left, outer_left)
        overlap_right = min(inner_right, outer_right)
        overlap_top = max(inner_top, outer_top)
        overlap_bottom = min(inner_bottom, outer_bottom)

        if overlap_right <= overlap_left or overlap_bottom <= overlap_top:
            return 0.0

        overlap_area = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
        inner_area = inner.width * inner.height

        if inner_area == 0:
            return 0.0

        return float(overlap_area / inner_area)

    def _edge_distance(self, a: BoundingBox, b: BoundingBox) -> float | None:
        """Calculate minimum edge-to-edge distance between bounds.

        Returns None if bounds overlap.
        """
        a_left = a.x
        a_right = a.x + a.width
        a_top = a.y
        a_bottom = a.y + a.height

        b_left = b.x
        b_right = b.x + b.width
        b_top = b.y
        b_bottom = b.y + b.height

        # Check for overlap
        if a_left < b_right and a_right > b_left and a_top < b_bottom and a_bottom > b_top:
            return None  # Overlapping

        # Calculate distances in each direction
        dx = 0.0
        dy = 0.0

        if a_right < b_left:
            dx = b_left - a_right
        elif b_right < a_left:
            dx = a_left - b_right

        if a_bottom < b_top:
            dy = b_top - a_bottom
        elif b_bottom < a_top:
            dy = a_top - b_bottom

        return float(np.sqrt(dx * dx + dy * dy))

    def _center_distance(self, a: BoundingBox, b: BoundingBox) -> float:
        """Calculate center-to-center distance."""
        a_center = (a.x + a.width / 2, a.y + a.height / 2)
        b_center = (b.x + b.width / 2, b.y + b.height / 2)

        dx = b_center[0] - a_center[0]
        dy = b_center[1] - a_center[1]

        return float(np.sqrt(dx * dx + dy * dy))

    def _check_alignment(self, a: BoundingBox, b: BoundingBox) -> str | None:
        """Check if two bounds are aligned.

        Returns:
            'horizontal', 'vertical', 'both', or None.
        """
        tolerance = self._alignment_tolerance

        # Horizontal alignment (same row)
        h_aligned = (
            abs(a.y - b.y) <= tolerance  # Top edges
            or abs((a.y + a.height) - (b.y + b.height)) <= tolerance  # Bottom edges
            or abs((a.y + a.height / 2) - (b.y + b.height / 2)) <= tolerance  # Centers
        )

        # Vertical alignment (same column)
        v_aligned = (
            abs(a.x - b.x) <= tolerance  # Left edges
            or abs((a.x + a.width) - (b.x + b.width)) <= tolerance  # Right edges
            or abs((a.x + a.width / 2) - (b.x + b.width / 2)) <= tolerance  # Centers
        )

        if h_aligned and v_aligned:
            return "both"
        elif h_aligned:
            return "horizontal"
        elif v_aligned:
            return "vertical"
        else:
            return None

    def _is_label_for(self, text_elem: Element, target: Element) -> bool:
        """Check if text element is a label for target.

        Labels are typically:
        - To the left of the element (same row)
        - Above the element (same column)
        - Within a reasonable distance
        """
        distance = self._center_distance(text_elem.bounds, target.bounds)
        if distance > self._label_distance_threshold:
            return False

        text_center_x = text_elem.bounds.x + text_elem.bounds.width / 2
        text_center_y = text_elem.bounds.y + text_elem.bounds.height / 2
        target_center_x = target.bounds.x + target.bounds.width / 2
        target_center_y = target.bounds.y + target.bounds.height / 2

        # Label to the left
        if (
            text_elem.bounds.x + text_elem.bounds.width < target.bounds.x
            and abs(text_center_y - target_center_y) < target.bounds.height
        ):
            return True

        # Label above
        if (
            text_elem.bounds.y + text_elem.bounds.height < target.bounds.y
            and abs(text_center_x - target_center_x) < target.bounds.width
        ):
            return True

        return False

    def _label_score(self, text_elem: Element, target: Element) -> float:
        """Calculate a score for label association."""
        distance = self._center_distance(text_elem.bounds, target.bounds)

        # Closer is better
        distance_score = max(0, 1 - (distance / self._label_distance_threshold))

        # Alignment bonus
        alignment = self._check_alignment(text_elem.bounds, target.bounds)
        alignment_bonus = 0.2 if alignment else 0.0

        return distance_score + alignment_bonus

    def _find_form_field_groups(
        self,
        elements: list[Element],
    ) -> list[ElementGroup]:
        """Find form field groups (label + input pairs)."""
        groups: list[ElementGroup] = []
        used_elements: set[str] = set()

        # Find text elements that could be labels
        text_elements = [e for e in elements if e.element_type == "text"]
        input_elements = [
            e for e in elements if e.element_type in ("input", "button", "checkbox", "dropdown")
        ]

        for input_elem in input_elements:
            if input_elem.id in used_elements:
                continue

            label = self.find_label_for(input_elem, text_elements)
            if label and label.id not in used_elements:
                # Create group
                group_elements = [label, input_elem]

                # Calculate group bounds
                min_x = min(e.bounds.x for e in group_elements)
                min_y = min(e.bounds.y for e in group_elements)
                max_x = max(e.bounds.x + e.bounds.width for e in group_elements)
                max_y = max(e.bounds.y + e.bounds.height for e in group_elements)

                group = ElementGroup(
                    id=f"form_field_{input_elem.id}",
                    elements=group_elements,
                    group_type="form_field",
                    bounds=BoundingBox(
                        x=min_x,
                        y=min_y,
                        width=max_x - min_x,
                        height=max_y - min_y,
                    ),
                    label=label.text,
                    confidence=0.8,
                    relationships=[
                        ElementRelationship(
                            source_id=label.id,
                            target_id=input_elem.id,
                            relationship_type=RelationshipType.LABELS,
                            confidence=0.8,
                        )
                    ],
                )

                groups.append(group)
                used_elements.add(label.id)
                used_elements.add(input_elem.id)

        return groups

    def _find_list_groups(
        self,
        elements: list[Element],
    ) -> list[ElementGroup]:
        """Find list groups (vertically stacked similar items)."""
        groups: list[ElementGroup] = []

        if len(elements) < 3:
            return groups

        # Sort by vertical position
        sorted_elements = sorted(elements, key=lambda e: e.bounds.y)

        # Look for vertically aligned elements with consistent spacing
        i = 0
        while i < len(sorted_elements) - 2:
            list_items = [sorted_elements[i]]
            base_elem = sorted_elements[i]

            # Check alignment tolerance
            for j in range(i + 1, len(sorted_elements)):
                candidate = sorted_elements[j]

                # Check vertical alignment (same column)
                if abs(candidate.bounds.x - base_elem.bounds.x) <= self._alignment_tolerance * 2:
                    # Check similar width
                    if (
                        abs(candidate.bounds.width - base_elem.bounds.width)
                        <= base_elem.bounds.width * 0.2
                    ):
                        list_items.append(candidate)

            # Need at least 3 items for a list
            if len(list_items) >= 3:
                # Check for consistent spacing
                spacings = []
                for k in range(1, len(list_items)):
                    spacing = list_items[k].bounds.y - (
                        list_items[k - 1].bounds.y + list_items[k - 1].bounds.height
                    )
                    spacings.append(spacing)

                if spacings:
                    avg_spacing = sum(spacings) / len(spacings)
                    spacing_variance = sum((s - avg_spacing) ** 2 for s in spacings) / len(spacings)

                    # Low variance means consistent spacing
                    if spacing_variance < avg_spacing * 0.5 or all(s >= 0 for s in spacings):
                        # Create group
                        min_x = min(e.bounds.x for e in list_items)
                        min_y = min(e.bounds.y for e in list_items)
                        max_x = max(e.bounds.x + e.bounds.width for e in list_items)
                        max_y = max(e.bounds.y + e.bounds.height for e in list_items)

                        group = ElementGroup(
                            id=f"list_{list_items[0].id}",
                            elements=list_items,
                            group_type="list",
                            bounds=BoundingBox(
                                x=min_x,
                                y=min_y,
                                width=max_x - min_x,
                                height=max_y - min_y,
                            ),
                            confidence=0.7,
                        )
                        groups.append(group)

                        # Skip processed elements
                        i += len(list_items)
                        continue

            i += 1

        return groups

    def _find_row_groups(
        self,
        elements: list[Element],
    ) -> list[ElementGroup]:
        """Find row groups (horizontally aligned items)."""
        groups: list[ElementGroup] = []

        if len(elements) < 2:
            return groups

        # Group by approximate Y position (horizontal bands)
        y_tolerance = 20
        rows: dict[int, list[Element]] = {}

        for elem in elements:
            center_y = elem.bounds.y + elem.bounds.height // 2
            row_key = center_y // y_tolerance

            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append(elem)

        # Create groups for rows with multiple elements
        for row_key, row_elements in rows.items():
            if len(row_elements) >= 2:
                # Sort by x position
                row_elements.sort(key=lambda e: e.bounds.x)

                min_x = min(e.bounds.x for e in row_elements)
                min_y = min(e.bounds.y for e in row_elements)
                max_x = max(e.bounds.x + e.bounds.width for e in row_elements)
                max_y = max(e.bounds.y + e.bounds.height for e in row_elements)

                group = ElementGroup(
                    id=f"row_{row_key}",
                    elements=row_elements,
                    group_type="row",
                    bounds=BoundingBox(
                        x=min_x,
                        y=min_y,
                        width=max_x - min_x,
                        height=max_y - min_y,
                    ),
                    confidence=0.6,
                )
                groups.append(group)

        return groups

    @classmethod
    def from_ocr_results(
        cls,
        ocr_results: list["OCRResult"],
        config: "VisionConfig | None" = None,
        environment: "GUIEnvironment | None" = None,
    ) -> tuple["RelationshipAnalyzer", list[Element]]:
        """Create analyzer and elements from OCR results.

        Args:
            ocr_results: OCR detection results.
            config: Vision configuration.
            environment: GUI environment.

        Returns:
            Tuple of (analyzer, elements).
        """
        analyzer = cls(config, environment)

        elements = [
            Element(
                id=f"text_{i}",
                bounds=result.bounds,
                element_type="text",
                text=result.text,
                confidence=result.confidence,
            )
            for i, result in enumerate(ocr_results)
        ]

        return analyzer, elements

    @classmethod
    def from_template_matches(
        cls,
        matches: list["TemplateMatch"],
        config: "VisionConfig | None" = None,
        environment: "GUIEnvironment | None" = None,
    ) -> tuple["RelationshipAnalyzer", list[Element]]:
        """Create analyzer and elements from template matches.

        Args:
            matches: Template matching results.
            config: Vision configuration.
            environment: GUI environment.

        Returns:
            Tuple of (analyzer, elements).
        """
        analyzer = cls(config, environment)

        elements = [
            Element(
                id=f"match_{i}",
                bounds=match.bounds,
                element_type="image",
                confidence=match.confidence,
                metadata={"template_name": getattr(match, "template_name", None)},
            )
            for i, match in enumerate(matches)
        ]

        return analyzer, elements


__all__ = [
    "Element",
    "ElementGroup",
    "ElementRelationship",
    "RelationshipAnalyzer",
    "RelationshipType",
]
