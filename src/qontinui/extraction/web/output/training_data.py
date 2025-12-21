"""
Training data exporter.

Exports extraction results to ML training data formats:
- COCO object detection format
- YOLO format
- JSONL (one annotation per line)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models import ElementType, ExtractionResult, StateType

logger = logging.getLogger(__name__)


class TrainingDataExporter:
    """Exports extraction results to ML training data formats."""

    # Map element types to COCO category IDs
    ELEMENT_CATEGORY_IDS: dict[ElementType, int] = {
        ElementType.BUTTON: 1,
        ElementType.TEXT_INPUT: 2,
        ElementType.PASSWORD_INPUT: 3,
        ElementType.TEXTAREA: 4,
        ElementType.LINK: 5,
        ElementType.DROPDOWN: 6,
        ElementType.CHECKBOX: 7,
        ElementType.RADIO: 8,
        ElementType.SLIDER: 9,
        ElementType.TOGGLE: 10,
        ElementType.TAB: 11,
        ElementType.MENU_ITEM: 12,
        ElementType.ICON_BUTTON: 13,
        ElementType.IMAGE: 14,
        ElementType.LABEL: 15,
        ElementType.HEADING: 16,
        ElementType.PARAGRAPH: 17,
        ElementType.LIST_ITEM: 18,
        ElementType.TABLE_CELL: 19,
        ElementType.UNKNOWN: 20,
    }

    # Map state types to COCO category IDs (offset by 100 to avoid collision)
    STATE_CATEGORY_IDS: dict[StateType, int] = {
        StateType.NAVIGATION: 101,
        StateType.MENU: 102,
        StateType.DROPDOWN_MENU: 103,
        StateType.DIALOG: 104,
        StateType.MODAL: 105,
        StateType.SIDEBAR: 106,
        StateType.TOOLBAR: 107,
        StateType.FORM: 108,
        StateType.CARD: 109,
        StateType.PANEL: 110,
        StateType.TOAST: 111,
        StateType.TOOLTIP: 112,
        StateType.POPOVER: 113,
        StateType.HEADER: 114,
        StateType.FOOTER: 115,
        StateType.CONTENT: 116,
        StateType.UNKNOWN: 117,
    }

    def export_coco(
        self,
        result: ExtractionResult,
        output_dir: Path,
        screenshots_dir: Path,
        include_states: bool = True,
    ) -> Path:
        """
        Export to COCO object detection format.

        Args:
            result: Extraction results.
            output_dir: Output directory.
            screenshots_dir: Directory containing screenshots.
            include_states: Whether to include state/region annotations.

        Returns:
            Path to the annotations JSON file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build COCO structure
        coco = {
            "info": {
                "description": "GUI Element Detection Dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat(),
                "extraction_id": result.extraction_id,
            },
            "licenses": [],
            "categories": self._build_categories(include_states),
            "images": [],
            "annotations": [],
        }

        annotation_id = 1
        image_id = 1

        # Group elements/states by screenshot
        elements_by_screenshot = self._group_by_screenshot(result)

        for screenshot_id, data in elements_by_screenshot.items():
            screenshot_path = screenshots_dir / f"{screenshot_id}.png"
            if not screenshot_path.exists():
                continue

            # Get image dimensions
            width, height = self._get_image_size(screenshot_path)

            # Add image entry
            coco["images"].append(  # type: ignore[attr-defined]
                {
                    "id": image_id,
                    "file_name": f"screenshots/{screenshot_id}.png",
                    "width": width,
                    "height": height,
                    "source_url": data.get("url", ""),
                }
            )

            # Add element annotations
            for element in data.get("elements", []):
                bbox = element.bbox
                coco["annotations"].append(  # type: ignore[attr-defined]
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": self.ELEMENT_CATEGORY_IDS.get(element.element_type, 20),
                        "bbox": [bbox.x, bbox.y, bbox.width, bbox.height],
                        "area": bbox.area,
                        "iscrowd": 0,
                        "attributes": {
                            "text_content": element.text_content,
                            "is_enabled": element.is_enabled,
                            "is_interactive": element.is_interactive,
                            "selector": element.selector,
                        },
                    }
                )
                annotation_id += 1

            # Add state annotations if requested
            if include_states:
                for state in data.get("states", []):
                    bbox = state.bbox
                    coco["annotations"].append(  # type: ignore[attr-defined]
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": self.STATE_CATEGORY_IDS.get(state.state_type, 117),
                            "bbox": [bbox.x, bbox.y, bbox.width, bbox.height],
                            "area": bbox.area,
                            "iscrowd": 0,
                            "attributes": {
                                "name": state.name,
                                "element_count": len(state.element_ids),
                            },
                        }
                    )
                    annotation_id += 1

            image_id += 1

        # Save annotations
        annotations_path = output_dir / "annotations.json"
        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)

        logger.info(f"Exported COCO format to {annotations_path}")
        return annotations_path

    def export_yolo(
        self,
        result: ExtractionResult,
        output_dir: Path,
        screenshots_dir: Path,
        include_states: bool = True,
    ) -> Path:
        """
        Export to YOLO format.

        Creates:
        - classes.txt with class names
        - labels/*.txt with annotations (one per image)
        - images/ directory (symlinks or copies)

        Args:
            result: Extraction results.
            output_dir: Output directory.
            screenshots_dir: Directory containing screenshots.
            include_states: Whether to include state/region annotations.

        Returns:
            Path to the output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        labels_dir = output_dir / "labels"
        labels_dir.mkdir(exist_ok=True)

        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Write classes.txt
        classes = self._build_yolo_classes(include_states)
        classes_path = output_dir / "classes.txt"
        with open(classes_path, "w") as f:
            f.write("\n".join(classes))

        # Group elements/states by screenshot
        elements_by_screenshot = self._group_by_screenshot(result)

        for screenshot_id, data in elements_by_screenshot.items():
            screenshot_path = screenshots_dir / f"{screenshot_id}.png"
            if not screenshot_path.exists():
                continue

            # Get image dimensions
            width, height = self._get_image_size(screenshot_path)

            # Build YOLO annotations
            lines = []

            for element in data.get("elements", []):
                class_id = (
                    self.ELEMENT_CATEGORY_IDS.get(element.element_type, 20) - 1
                )  # YOLO is 0-indexed
                bbox = element.bbox

                # Convert to YOLO format (center_x, center_y, width, height) normalized
                cx = (bbox.x + bbox.width / 2) / width
                cy = (bbox.y + bbox.height / 2) / height
                w = bbox.width / width
                h = bbox.height / height

                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if include_states:
                for state in data.get("states", []):
                    class_id = self.STATE_CATEGORY_IDS.get(state.state_type, 117) - 1
                    bbox = state.bbox

                    cx = (bbox.x + bbox.width / 2) / width
                    cy = (bbox.y + bbox.height / 2) / height
                    w = bbox.width / width
                    h = bbox.height / height

                    lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # Write label file
            label_path = labels_dir / f"{screenshot_id}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(lines))

            # Symlink or copy image
            dest_image = images_dir / f"{screenshot_id}.png"
            if not dest_image.exists():
                try:
                    dest_image.symlink_to(screenshot_path.resolve())
                except OSError:
                    import shutil

                    shutil.copy2(screenshot_path, dest_image)

        logger.info(f"Exported YOLO format to {output_dir}")
        return output_dir

    def export_jsonl(
        self,
        result: ExtractionResult,
        output_path: Path,
        include_states: bool = True,
    ) -> Path:
        """
        Export to JSONL format (one annotation per line).

        Args:
            result: Extraction results.
            output_path: Output file path.
            include_states: Whether to include state annotations.

        Returns:
            Path to the output file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        elements_by_screenshot = self._group_by_screenshot(result)

        with open(output_path, "w", encoding="utf-8") as f:
            for screenshot_id, data in elements_by_screenshot.items():
                annotation = {
                    "screenshot_id": screenshot_id,
                    "url": data.get("url", ""),
                    "viewport": data.get("viewport", []),
                    "elements": [
                        {
                            "id": e.id,
                            "bbox": e.bbox.to_dict(),
                            "element_type": e.element_type.value,
                            "text_content": e.text_content,
                            "is_interactive": e.is_interactive,
                            "is_enabled": e.is_enabled,
                            "selector": e.selector,
                        }
                        for e in data.get("elements", [])
                    ],
                }

                if include_states:
                    annotation["states"] = [
                        {
                            "id": s.id,
                            "name": s.name,
                            "bbox": s.bbox.to_dict(),
                            "state_type": s.state_type.value,
                            "element_ids": s.element_ids,
                        }
                        for s in data.get("states", [])
                    ]

                f.write(json.dumps(annotation, ensure_ascii=False) + "\n")

        logger.info(f"Exported JSONL format to {output_path}")
        return output_path

    def _build_categories(self, include_states: bool) -> list[dict[str, Any]]:
        """Build COCO categories list."""
        categories = []

        # Element categories
        for element_type, cat_id in self.ELEMENT_CATEGORY_IDS.items():
            categories.append(
                {
                    "id": cat_id,
                    "name": element_type.value,
                    "supercategory": "element",
                }
            )

        # State categories
        if include_states:
            for state_type, cat_id in self.STATE_CATEGORY_IDS.items():
                categories.append(
                    {
                        "id": cat_id,
                        "name": state_type.value,
                        "supercategory": "state",
                    }
                )

        return categories

    def _build_yolo_classes(self, include_states: bool) -> list[str]:
        """Build YOLO classes list."""
        classes = [et.value for et in ElementType]
        if include_states:
            classes.extend([st.value for st in StateType])
        return classes

    def _group_by_screenshot(self, result: ExtractionResult) -> dict[str, dict[str, Any]]:
        """Group elements and states by their screenshot ID."""
        grouped: dict[str, dict[str, Any]] = {}

        # Get page info for each screenshot
        for page in result.page_extractions:
            for screenshot_id in page.screenshot_ids:
                if screenshot_id not in grouped:
                    grouped[screenshot_id] = {
                        "url": page.url,
                        "viewport": list(page.viewport),
                        "elements": [],
                        "states": [],
                    }

                # Add elements from this page
                grouped[screenshot_id]["elements"].extend(page.elements)
                grouped[screenshot_id]["states"].extend(page.states)

        return grouped

    def _get_image_size(self, image_path: Path) -> tuple[int, int]:
        """Get image dimensions."""
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                return img.size  # type: ignore[no-any-return]
        except Exception:
            return (1920, 1080)  # Default fallback
