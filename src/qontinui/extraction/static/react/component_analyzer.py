"""
Component Analyzer for React.

Handles component extraction and classification.
"""

from __future__ import annotations

import logging
from pathlib import Path

from qontinui.extraction.static.models import ComponentCategory, ComponentDefinition

from . import components as comp_module

logger = logging.getLogger(__name__)


class ComponentAnalyzer:
    """Analyzer for extracting and classifying React components."""

    def __init__(self):
        """Initialize the component analyzer."""
        self.components: list[ComponentDefinition] = []

    def extract_components(self, parse_result: dict, file_path: Path) -> list[ComponentDefinition]:
        """
        Extract all components from a parse result.

        Args:
            parse_result: TypeScript parse result
            file_path: Source file path

        Returns:
            List of ComponentDefinition objects
        """
        function_components = comp_module.extract_function_components(parse_result, file_path)
        class_components = comp_module.extract_class_components(parse_result, file_path)

        all_components = function_components + class_components
        self.components.extend(all_components)
        return all_components

    def classify_components(self) -> None:
        """Classify components as states (page-level) or widgets (UI elements)."""
        comp_module.classify_components(self.components)

    def build_relationships(self) -> None:
        """Build component parent-child relationships."""
        # This would require access to JSX rendering information
        # For now, we'll leave component hierarchies to be built separately
        pass

    def get_components(self) -> list[ComponentDefinition]:
        """Get all extracted components."""
        return self.components

    def reset(self) -> None:
        """Reset the analyzer state."""
        self.components = []

    def get_stats(self) -> dict[str, int]:
        """
        Get statistics about extracted components.

        Returns:
            Dictionary with component counts by category
        """
        state_count = sum(1 for c in self.components if c.category == ComponentCategory.STATE)
        widget_count = sum(1 for c in self.components if c.category == ComponentCategory.WIDGET)

        return {
            "total": len(self.components),
            "states": state_count,
            "widgets": widget_count,
        }
