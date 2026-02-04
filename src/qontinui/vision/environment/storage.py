"""Storage utilities for GUI environment data.

Provides functions for saving and loading GUIEnvironment models to/from
JSON files with proper serialization handling.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from qontinui_schemas.testing.environment import GUIEnvironment

logger = logging.getLogger(__name__)


def _serialize_datetime(obj: Any) -> Any:
    """Serialize datetime objects to ISO format strings.

    Args:
        obj: Object to serialize.

    Returns:
        Serialized object.

    Raises:
        TypeError: If object is not JSON serializable.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_environment(
    environment: GUIEnvironment,
    file_path: str | Path,
    indent: int = 2,
) -> None:
    """Save a GUIEnvironment model to a JSON file.

    Args:
        environment: The GUIEnvironment model to save.
        file_path: Path to the output JSON file.
        indent: JSON indentation level (default: 2).

    Raises:
        IOError: If the file cannot be written.
        ValueError: If the environment cannot be serialized.
    """
    file_path = Path(file_path)

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Use model_dump() for Pydantic v2
        data = environment.model_dump(mode="json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=_serialize_datetime)

        logger.info(f"Saved GUI environment to {file_path}")

    except Exception as e:
        logger.error(f"Failed to save environment to {file_path}: {e}")
        raise


def load_environment(file_path: str | Path) -> GUIEnvironment:
    """Load a GUIEnvironment model from a JSON file.

    Args:
        file_path: Path to the input JSON file.

    Returns:
        Loaded GUIEnvironment model.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contents are invalid.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Environment file not found: {file_path}")

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        environment = GUIEnvironment.model_validate(data)
        logger.info(f"Loaded GUI environment from {file_path}")

        return environment

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in environment file {file_path}: {e}")
        raise ValueError(f"Invalid JSON format: {e}") from e

    except Exception as e:
        logger.error(f"Failed to load environment from {file_path}: {e}")
        raise


def export_environment_summary(
    environment: GUIEnvironment,
    file_path: str | Path,
) -> None:
    """Export a human-readable summary of the environment.

    Creates a markdown file with key findings from the environment analysis.

    Args:
        environment: The GUIEnvironment model to summarize.
        file_path: Path to the output markdown file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# GUI Environment Summary",
        "",
        f"**App Identifier:** {environment.app_identifier or 'Unknown'}",
        f"**Discovery Date:** {environment.discovery_timestamp or 'N/A'}",
        f"**Screenshots Analyzed:** {environment.screenshots_analyzed}",
        f"**Actions Observed:** {environment.actions_observed}",
        "",
        "## Confidence Scores",
        "",
        f"- Color Extraction: {environment.confidence_scores.color_extraction:.1%}",
        f"- Typography Detection: {environment.confidence_scores.typography_detection:.1%}",
        f"- Layout Analysis: {environment.confidence_scores.layout_analysis:.1%}",
        f"- Dynamic Detection: {environment.confidence_scores.dynamic_detection:.1%}",
        f"- State Learning: {environment.confidence_scores.state_learning:.1%}",
        f"- Element Detection: {environment.confidence_scores.element_detection:.1%}",
        "",
        "## Color Palette",
        "",
        f"**Theme:** {environment.colors.theme_type.value}",
        f"**Dominant Colors:** {', '.join(environment.colors.dominant_colors[:5]) or 'None detected'}",
        "",
    ]

    # Add semantic colors if detected
    semantic = environment.colors.semantic_colors
    if any([semantic.background, semantic.accent, semantic.error]):
        lines.extend(
            [
                "**Semantic Colors:**",
                f"- Background: {semantic.background or 'N/A'}",
                f"- Accent: {semantic.accent or 'N/A'}",
                f"- Error: {semantic.error or 'N/A'}",
                f"- Success: {semantic.success or 'N/A'}",
                "",
            ]
        )

    # Typography
    lines.extend(
        [
            "## Typography",
            "",
            f"**Fonts Detected:** {len(environment.typography.detected_fonts)}",
            f"**Languages:** {', '.join(environment.typography.languages_detected) or 'N/A'}",
            "",
        ]
    )

    sizes = environment.typography.text_sizes
    if sizes.body:
        lines.extend(
            [
                "**Text Sizes:**",
                f"- Body: {sizes.body}px",
                f"- Heading: {sizes.heading or 'N/A'}px",
                f"- Small: {sizes.small or 'N/A'}px",
                "",
            ]
        )

    # Layout
    lines.extend(
        [
            "## Layout",
            "",
            f"**Regions Detected:** {len(environment.layout.regions)}",
            f"**Grid Detected:** {'Yes' if environment.layout.grid.detected else 'No'}",
            "",
        ]
    )

    if environment.layout.regions:
        lines.append("**Regions:**")
        for name, region in environment.layout.regions.items():
            lines.append(
                f"- {name}: {region.semantic_label.value} "
                f"({region.bounds.width}x{region.bounds.height})"
            )
        lines.append("")

    # Dynamic regions
    lines.extend(
        [
            "## Dynamic Regions",
            "",
            f"**Always Changing:** {len(environment.dynamic_regions.always_changing)}",
            f"**Conditional:** {len(environment.dynamic_regions.conditionally_changing)}",
            f"**Animations:** {len(environment.dynamic_regions.animation_regions)}",
            "",
        ]
    )

    # Visual states
    lines.extend(
        [
            "## Visual States",
            "",
            f"**Element Types Learned:** {len(environment.visual_states.element_states)}",
            "",
        ]
    )

    for elem_type, states in environment.visual_states.element_states.items():
        state_names = list(states.states.keys())
        lines.append(f"- {elem_type}: {', '.join(state_names)}")

    lines.append("")

    # Element patterns
    lines.extend(
        [
            "## Element Patterns",
            "",
            f"**Patterns Detected:** {len(environment.element_patterns.patterns)}",
            f"**Total Elements:** {environment.element_patterns.elements_detected}",
            "",
        ]
    )

    for elem_type, pattern in environment.element_patterns.patterns.items():
        lines.append(f"- {elem_type}: {pattern.shape.value}, {pattern.detection_count} instances")

    lines.append("")

    # Write file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Exported environment summary to {file_path}")


__all__ = [
    "save_environment",
    "load_environment",
    "export_environment_summary",
]
