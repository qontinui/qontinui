"""Spatial relationship analysis for UI elements.

This module provides specialized spatial analysis capabilities for identifying
alignment, adjacency, containment, and other spatial relationships between
UI elements.
"""

import numpy as np

from qontinui.discovery.state_construction.element_identifier import (
    IdentifiedElement,
    SpatialRelationship,
)


class SpatialAnalyzer:
    """Analyzes spatial relationships between UI elements.

    Identifies alignment, adjacency, containment, and other spatial
    relationships that help understand UI structure and layout patterns.
    """

    def __init__(
        self,
        alignment_tolerance: int = 5,
        adjacency_tolerance: int = 10,
    ):
        """Initialize the spatial analyzer.

        Args:
            alignment_tolerance: Pixel tolerance for alignment detection
            adjacency_tolerance: Pixel tolerance for adjacency detection
        """
        self.alignment_tolerance = alignment_tolerance
        self.adjacency_tolerance = adjacency_tolerance

    def analyze_spatial_layout(
        self, elements: list[IdentifiedElement]
    ) -> list[SpatialRelationship]:
        """Analyze spatial relationships between elements.

        Identifies alignment, adjacency, containment, and other spatial
        relationships that help understand UI structure.

        Args:
            elements: List of identified elements

        Returns:
            List of spatial relationships between elements
        """
        relationships = []

        for i, elem1 in enumerate(elements):
            for _j, elem2 in enumerate(elements[i + 1 :], start=i + 1):
                # Check for various relationship types
                rel = self._analyze_element_pair(elem1, elem2)
                if rel:
                    relationships.append(rel)

        return relationships

    def _analyze_element_pair(
        self, elem1: IdentifiedElement, elem2: IdentifiedElement
    ) -> SpatialRelationship | None:
        """Analyze spatial relationship between two elements.

        Args:
            elem1: First element
            elem2: Second element

        Returns:
            SpatialRelationship if significant relationship found
        """
        x1, y1, w1, h1 = elem1.bounds
        x2, y2, w2, h2 = elem2.bounds

        # Calculate distance between centers
        center1 = (x1 + w1 / 2, y1 + h1 / 2)
        center2 = (x2 + w2 / 2, y2 + h2 / 2)
        distance = np.sqrt((center2[0] - center1[0]) ** 2 + (center2[1] - center1[1]) ** 2)

        properties = {}
        relationship = "none"

        # Check alignment
        if abs(y1 - y2) <= self.alignment_tolerance:
            relationship = "horizontally_aligned"
            properties["alignment"] = "horizontal"
        elif abs(x1 - x2) <= self.alignment_tolerance:
            relationship = "vertically_aligned"
            properties["alignment"] = "vertical"

        # Check adjacency
        if abs((x1 + w1) - x2) <= self.adjacency_tolerance:
            relationship = "adjacent_left"
            properties["adjacency"] = "left"
        elif abs(x1 - (x2 + w2)) <= self.adjacency_tolerance:
            relationship = "adjacent_right"
            properties["adjacency"] = "right"
        elif abs((y1 + h1) - y2) <= self.adjacency_tolerance:
            relationship = "adjacent_above"
            properties["adjacency"] = "above"
        elif abs(y1 - (y2 + h2)) <= self.adjacency_tolerance:
            relationship = "adjacent_below"
            properties["adjacency"] = "below"

        # Check containment
        if x1 <= x2 and y1 <= y2 and x1 + w1 >= x2 + w2 and y1 + h1 >= y2 + h2:
            relationship = "contains"
            properties["containment"] = "contains"
        elif x2 <= x1 and y2 <= y1 and x2 + w2 >= x1 + w1 and y2 + h2 >= y1 + h1:
            relationship = "contained_by"
            properties["containment"] = "contained_by"

        # Only return if we found a meaningful relationship
        if relationship != "none":
            # Generate IDs from element bounds (since IdentifiedElement doesn't have ID)
            elem1_id = f"elem_{x1}_{y1}_{w1}_{h1}"
            elem2_id = f"elem_{x2}_{y2}_{w2}_{h2}"

            return SpatialRelationship(
                element1_id=elem1_id,
                element2_id=elem2_id,
                relationship=relationship,
                distance=float(distance),
                properties=properties,
            )

        return None
