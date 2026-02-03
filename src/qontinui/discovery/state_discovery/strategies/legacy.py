"""Legacy ID-based state discovery strategy.

This strategy uses element IDs (data-ui-id, data-testid, html id) and
co-occurrence analysis to discover states. It is the original implementation
migrated from ui_bridge_adapter.py.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from qontinui.discovery.models import StateImage
from qontinui.discovery.pixel_analysis.analyzers.cooccurrence_analyzer import CooccurrenceAnalyzer

from ..base import (
    DiscoveredElement,
    DiscoveredState,
    DiscoveryStrategyType,
    StateDiscoveryInput,
    StateDiscoveryResult,
    StateDiscoveryStrategy,
)

logger = logging.getLogger(__name__)


class LegacyStrategy(StateDiscoveryStrategy):
    """ID-based state discovery using co-occurrence analysis.

    This strategy:
    1. Extracts element IDs from render snapshots
    2. Builds element -> render mapping
    3. Groups elements by co-occurrence (same render membership)
    4. Returns discovered states
    """

    @property
    def strategy_type(self) -> DiscoveryStrategyType:
        return DiscoveryStrategyType.LEGACY

    def can_process(self, input_data: StateDiscoveryInput) -> bool:
        """Check if we have render data to process."""
        return input_data.has_render_data()

    def discover(self, input_data: StateDiscoveryInput) -> StateDiscoveryResult:
        """Discover states from render snapshots.

        Args:
            input_data: Input containing renders list

        Returns:
            Discovery result with states and elements
        """
        renders = input_data.renders
        include_html_ids = input_data.include_html_ids

        if not renders:
            return StateDiscoveryResult(
                states=[],
                elements=[],
                element_to_renders={},
                render_count=0,
                unique_element_count=0,
                strategy_used=self.strategy_type,
            )

        # Step 1: Build element -> render_ids mapping
        element_to_renders, render_elements = self._build_element_render_mapping(
            renders, include_html_ids
        )

        if not element_to_renders:
            return StateDiscoveryResult(
                states=[],
                elements=[],
                element_to_renders={},
                render_count=len(renders),
                unique_element_count=0,
                strategy_used=self.strategy_type,
            )

        # Step 2: Convert to StateImage format for co-occurrence algorithm
        state_images = self._create_state_images(element_to_renders, len(renders))

        # Step 3: Run co-occurrence analysis
        discovered_states = self._run_cooccurrence_analysis(state_images, len(renders))

        # Step 4: Build element list
        elements = self._build_element_list(element_to_renders)

        # Step 5: Convert discovered states to unified format
        unified_states = self._convert_states(discovered_states)

        # Convert set values to lists for serialization
        element_to_renders_serializable = {k: sorted(v) for k, v in element_to_renders.items()}

        logger.info(
            f"Legacy strategy discovered {len(unified_states)} states "
            f"from {len(renders)} renders with {len(elements)} elements"
        )

        return StateDiscoveryResult(
            states=unified_states,
            elements=elements,
            element_to_renders=element_to_renders_serializable,
            render_count=len(renders),
            unique_element_count=len(element_to_renders),
            strategy_used=self.strategy_type,
            strategy_metadata={"algorithm": "cooccurrence"},
        )

    def _build_element_render_mapping(
        self,
        renders: list[dict[str, Any]],
        include_html_ids: bool = False,
    ) -> tuple[dict[str, set[str]], list[tuple[str, list[str]]]]:
        """Build mapping of elements to the renders they appear in."""
        element_to_renders: dict[str, set[str]] = {}
        render_elements: list[tuple[str, list[str]]] = []

        for i, render in enumerate(renders):
            render_id = render.get("id", f"render_{i}")
            elements = self._extract_elements_from_render(render, include_html_ids)
            render_elements.append((render_id, elements))

            for elem_id in elements:
                if elem_id not in element_to_renders:
                    element_to_renders[elem_id] = set()
                element_to_renders[elem_id].add(render_id)

        return element_to_renders, render_elements

    def _extract_elements_from_render(
        self,
        render_log_entry: dict[str, Any],
        include_html_ids: bool = False,
    ) -> list[str]:
        """Extract element IDs from a UI Bridge render log entry."""
        element_ids: set[str] = set()

        # Format 1: DomSnapshotRenderLogEntry (from qontinui-web)
        if render_log_entry.get("type") == "dom_snapshot":
            snapshot = render_log_entry.get("snapshot", {})
            root = snapshot.get("root")
            if root:
                self._extract_from_dom_node(root, element_ids, include_html_ids)
            return sorted(element_ids)

        # Format 2: Simple format (for testing and backward compatibility)
        if "elements" in render_log_entry:
            for elem in render_log_entry["elements"]:
                elem_id = elem.get("id")
                if elem_id:
                    element_ids.add(f"reg:{elem_id}")

        # Extract from componentTree (simple format)
        if "componentTree" in render_log_entry:
            self._extract_from_dom_node(
                render_log_entry["componentTree"], element_ids, include_html_ids
            )

        # Also check "tree" key (alternative simple format)
        if "tree" in render_log_entry:
            self._extract_from_dom_node(render_log_entry["tree"], element_ids, include_html_ids)

        return sorted(element_ids)

    def _extract_from_dom_node(
        self,
        node: dict[str, Any],
        element_ids: set[str],
        include_html_ids: bool = False,
    ) -> None:
        """Recursively extract element IDs from a DOM node."""
        if not isinstance(node, dict):
            return

        # Get attributes dict
        attrs = node.get("attributes", {})
        if isinstance(attrs, dict):
            # Priority 1: data-ui-id (UI Bridge registered elements)
            ui_id = attrs.get("data-ui-id")
            if ui_id:
                element_ids.add(f"ui:{ui_id}")

            # Priority 2: data-testid (testing convention)
            testid = attrs.get("data-testid")
            if testid:
                element_ids.add(f"testid:{testid}")

            # Optional: HTML id attribute
            if include_html_ids:
                html_id = node.get("id") or attrs.get("id")
                if html_id:
                    element_ids.add(f"html:{html_id}")

        # Also check props (for React component tree format)
        props = node.get("props", {})
        if isinstance(props, dict):
            ui_id = props.get("data-ui-id")
            if ui_id:
                element_ids.add(f"ui:{ui_id}")

            testid = props.get("data-testid")
            if testid:
                element_ids.add(f"testid:{testid}")

        # Recurse into children
        children = node.get("children", [])
        if isinstance(children, list):
            for child in children:
                self._extract_from_dom_node(child, element_ids, include_html_ids)

    def _create_state_images(
        self, element_to_renders: dict[str, set[str]], render_count: int
    ) -> list[StateImage]:
        """Convert element mapping to StateImage format for algorithm."""
        state_images: list[StateImage] = []

        for elem_id, render_ids in element_to_renders.items():
            state_images.append(
                StateImage(
                    id=elem_id,
                    name=elem_id,
                    x=0,
                    y=0,
                    x2=1,
                    y2=1,  # Placeholder coordinates
                    pixel_hash="",  # Not used for UI Bridge
                    frequency=len(render_ids) / render_count,
                    screenshot_ids=sorted(render_ids),
                )
            )

        return state_images

    def _run_cooccurrence_analysis(
        self, state_images: list[StateImage], render_count: int
    ) -> list[Any]:
        """Run co-occurrence analysis on state images."""
        # Create dummy screenshot array (algorithm normalizes by length)
        dummy_screenshots = [np.zeros((1, 1, 3), dtype=np.uint8) for _ in range(render_count)]

        analyzer = CooccurrenceAnalyzer()
        return analyzer.analyze(state_images, dummy_screenshots)

    def _build_element_list(
        self, element_to_renders: dict[str, set[str]]
    ) -> list[DiscoveredElement]:
        """Build list of discovered elements."""
        elements: list[DiscoveredElement] = []

        for elem_id, render_ids in element_to_renders.items():
            # Determine element type from prefix
            if elem_id.startswith("ui:"):
                elem_type = "ui-id"
            elif elem_id.startswith("testid:"):
                elem_type = "testid"
            elif elem_id.startswith("html:"):
                elem_type = "html-id"
            elif elem_id.startswith("reg:"):
                elem_type = "registered"
            else:
                elem_type = "unknown"

            clean_name = elem_id.split(":", 1)[1] if ":" in elem_id else elem_id

            elements.append(
                DiscoveredElement(
                    id=elem_id,
                    name=clean_name,
                    element_type=elem_type,
                    render_ids=sorted(render_ids),
                )
            )

        return elements

    def _convert_states(self, discovered_states: list[Any]) -> list[DiscoveredState]:
        """Convert internal discovered states to unified format."""
        unified_states: list[DiscoveredState] = []

        for state in discovered_states:
            # Determine if state is blocking (modal dialogs, etc.)
            is_modal = False
            modal_indicators = ["modal", "dialog", "popup", "alert", "overlay"]
            state_name_lower = state.name.lower()

            for indicator in modal_indicators:
                if indicator in state_name_lower:
                    is_modal = True
                    break

            unified_states.append(
                DiscoveredState(
                    id=state.id,
                    name=state.name,
                    element_ids=state.state_image_ids,
                    render_ids=state.screenshot_ids,
                    confidence=state.confidence,
                    is_modal=is_modal,
                    metadata=state.metadata,
                )
            )

        return unified_states
