"""Shared modules for discovery features.

This package contains components shared between different discovery features:
- Extraction (StateImage discovery from screenshots)
- Input Capture (click-to-template workflow)

Shared Components:
    CoOccurrenceGrouper: Groups images/templates by co-occurrence patterns
    invert_frame_template_map: Converts frame→templates to template→frames
    from_state_images_screenshot_ids: Converts StateImages to common format
"""

from .co_occurrence_grouper import (
    CoOccurrenceGroup,
    CoOccurrenceGrouper,
    CoOccurrenceGroupingResult,
    from_state_images_screenshot_ids,
    invert_frame_template_map,
)

__all__ = [
    "CoOccurrenceGrouper",
    "CoOccurrenceGroup",
    "CoOccurrenceGroupingResult",
    "invert_frame_template_map",
    "from_state_images_screenshot_ids",
]
