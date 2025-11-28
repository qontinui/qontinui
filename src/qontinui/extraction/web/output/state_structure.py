"""
State structure exporter.

Exports extraction results to Qontinui's native state structure format.
"""

import json
import logging
from pathlib import Path
from typing import Any

from ..models import (
    ExtractedElement,
    ExtractedState,
    ExtractedTransition,
    ExtractionResult,
)

logger = logging.getLogger(__name__)


class StateStructureExporter:
    """Exports extraction results to Qontinui state structure format."""

    def __init__(self, include_screenshots: bool = True):
        """
        Initialize exporter.

        Args:
            include_screenshots: Whether to copy screenshots to output directory.
        """
        self.include_screenshots = include_screenshots

    def export(
        self,
        result: ExtractionResult,
        output_dir: Path,
        screenshots_dir: Path | None = None,
    ) -> Path:
        """
        Export extraction results to Qontinui format.

        Args:
            result: Extraction results to export.
            output_dir: Directory to save output files.
            screenshots_dir: Source directory for screenshots (if copying).

        Returns:
            Path to the exported state structure JSON file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create images directory
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Create states directory
        states_dir = output_dir / "states"
        states_dir.mkdir(exist_ok=True)

        # Copy screenshots if requested
        if self.include_screenshots and screenshots_dir:
            self._copy_screenshots(screenshots_dir, images_dir, result)

        # Build state structure
        structure = self._build_structure(result, images_dir)

        # Save JSON
        output_path = output_dir / "state_structure.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(structure, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported state structure to {output_path}")
        return output_path

    def _build_structure(
        self,
        result: ExtractionResult,
        images_dir: Path,
    ) -> dict[str, Any]:
        """Build the state structure dictionary."""
        return {
            "version": "1.0",
            "extraction_id": result.extraction_id,
            "source_urls": result.source_urls,
            "extracted_at": result.started_at.isoformat(),
            "states": [self._export_state(s, images_dir) for s in result.states],
            "transitions": [self._export_transition(t) for t in result.transitions],
            "images": [
                self._export_element_as_image(e, images_dir)
                for e in result.elements
                if e.is_interactive
            ],
            "metadata": {
                "viewports": [list(v) for v in result.viewports],
                "config": result.config,
                "statistics": {
                    "total_states": len(result.states),
                    "total_elements": len(result.elements),
                    "total_transitions": len(result.transitions),
                    "total_pages": len(result.page_extractions),
                },
            },
        }

    def _export_state(self, state: ExtractedState, images_dir: Path) -> dict[str, Any]:
        """Export a state to Qontinui format."""
        return {
            "id": state.id,
            "name": state.name,
            "type": state.state_type.value,
            "detection_method": state.detection_method,
            "confidence": state.confidence,
            "bounding_box": state.bbox.to_dict(),
            "element_ids": state.element_ids,
            "screenshot": f"states/{state.screenshot_id}.png",
            "semantic_role": state.semantic_role,
            "aria_label": state.aria_label,
            "source_url": state.source_url,
            "metadata": state.metadata,
        }

    def _export_transition(self, transition: ExtractedTransition) -> dict[str, Any]:
        """Export a transition to Qontinui format."""
        return {
            "id": transition.id,
            "action_type": transition.action_type.value,
            "target_element_id": transition.target_element_id,
            "target_selector": transition.target_selector,
            "causes_appear": transition.causes_appear,
            "causes_disappear": transition.causes_disappear,
            "action_value": transition.action_value,
            "key_modifiers": transition.key_modifiers,
            "confidence": transition.confidence,
            "metadata": transition.metadata,
        }

    def _export_element_as_image(
        self, element: ExtractedElement, images_dir: Path
    ) -> dict[str, Any]:
        """Export an element as a Qontinui image object."""
        return {
            "id": element.id,
            "name": element.name or element.text_content or element.element_type.value,
            "type": element.element_type.value,
            "bounding_box": element.bbox.to_dict(),
            "selector": element.selector,
            "is_interactive": element.is_interactive,
            "is_enabled": element.is_enabled,
            "text_content": element.text_content,
            "semantic_role": element.semantic_role,
            "aria_label": element.aria_label,
            "attributes": {k: v for k, v in element.attributes.items() if v is not None},
        }

    def _copy_screenshots(
        self,
        source_dir: Path,
        dest_dir: Path,
        result: ExtractionResult,
    ) -> None:
        """Copy relevant screenshots to output directory."""
        import shutil

        for screenshot_id in result.screenshot_ids:
            source_file = source_dir / f"{screenshot_id}.png"
            if source_file.exists():
                dest_file = dest_dir / f"{screenshot_id}.png"
                shutil.copy2(source_file, dest_file)


def to_qontinui_state_config(state: ExtractedState) -> dict[str, Any]:
    """
    Convert an extracted state to Qontinui's internal State configuration.

    This can be used to create State objects programmatically.
    """
    return {
        "name": state.name,
        "description": f"Extracted from {state.source_url}",
        "state_images": [],  # Would need actual image data
        "state_regions": [
            {
                "name": f"{state.name}_region",
                "x": state.bbox.x,
                "y": state.bbox.y,
                "width": state.bbox.width,
                "height": state.bbox.height,
            }
        ],
        "metadata": {
            "extraction_id": state.id,
            "state_type": state.state_type.value,
            "detection_method": state.detection_method,
            "confidence": state.confidence,
        },
    }
