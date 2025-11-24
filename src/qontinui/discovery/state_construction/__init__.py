"""State construction module for building state objects from detected elements.

This module provides functionality for constructing complete state representations
from detected elements and regions, including defining state properties, identifying
defining features, and establishing relationships between states.

State construction transforms raw detection data into structured state objects
that can be used for navigation, testing, and automation workflows.

Key Components:
    - State object creation from detected elements
    - Defining feature identification
    - State relationship mapping
    - State validation and refinement
    - State merging and deduplication
    - State metadata enrichment

Example:
    >>> from qontinui.discovery.state_construction import construct_state
    >>> elements = detect_elements(screenshot)
    >>> state = construct_state(elements, screenshot)
    >>> print(f"Constructed state '{state.name}' with {len(state.elements)} elements")
"""

from .element_identifier import (
    ElementIdentifier,
    ElementType,
    IdentifiedElement,
    IdentifiedRegion,
    RegionType,
    SpatialRelationship,
)
from .ocr_name_generator import (
    NameValidator,
    OCRNameGenerator,
    generate_element_name,
    generate_state_name_from_screenshot,
)
from .state_builder import (
    FallbackElementIdentifier,
    FallbackNameGenerator,
    StateBuilder,
    TransitionInfo,
)

__all__ = [
    "ElementIdentifier",
    "ElementType",
    "RegionType",
    "IdentifiedElement",
    "IdentifiedRegion",
    "SpatialRelationship",
    "OCRNameGenerator",
    "NameValidator",
    "generate_element_name",
    "generate_state_name_from_screenshot",
    "StateBuilder",
    "TransitionInfo",
    "FallbackNameGenerator",
    "FallbackElementIdentifier",
]

__version__ = "0.1.0"
