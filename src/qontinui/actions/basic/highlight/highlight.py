"""Highlight action implementation - ported from Qontinui framework.

Highlights regions on screen for debugging and visualization.
"""

import time
from typing import Any

from ....model.element.region import Region
from ...action_interface import ActionInterface
from ...action_result import ActionResult
from ...object_collection import ObjectCollection
from ..find.find import Find
from .highlight_options import HighlightOptions


class Highlight(ActionInterface):
    """Highlights regions on screen for debugging and visualization.

    Port of Highlight from Qontinui framework action.

    The Highlight action is primarily used for:
    - Debugging pattern matching results
    - Providing visual feedback during automation
    - Demonstrating where actions will occur
    """

    def __init__(self, find: Find | None = None) -> None:
        """Initialize Highlight action.

        Args:
            find: The Find action for locating regions to highlight
        """
        self.find = find

    def perform(
        self, action_result: ActionResult, *object_collections: ObjectCollection
    ) -> None:
        """Execute the highlight operation.

        Finds target regions and displays visual highlights around them.

        Args:
            action_result: Result container that will be populated
            object_collections: Collections defining what to highlight
        """
        if not isinstance(action_result.action_config, HighlightOptions):
            object.__setattr__(action_result, "success", False)
            return

        highlight_options = action_result.action_config

        # Find regions to highlight
        regions = self._find_regions_to_highlight(action_result, object_collections)
        if not regions:
            object.__setattr__(action_result, "success", False)
            object.__setattr__(
                action_result, "output_text", "No regions found to highlight"
            )
            return

        # Perform highlighting
        success = self._highlight_regions(
            regions,
            highlight_options.get_color(),
            highlight_options.get_thickness(),
            highlight_options.get_highlight_duration(),
            highlight_options.is_flash(),
            highlight_options.get_flash_times(),
        )

        object.__setattr__(action_result, "success", success)
        if success:
            object.__setattr__(
                action_result, "output_text", f"Highlighted {len(regions)} region(s)"
            )

    def _find_regions_to_highlight(
        self, action_result: ActionResult, object_collections: tuple[Any, ...]
    ) -> list[Region]:
        """Find regions to highlight.

        Args:
            action_result: The action result for storing find results
            object_collections: Collections to search in

        Returns:
            List of regions to highlight
        """
        regions: list[Region] = []

        if not object_collections:
            return regions

        # Use Find to locate targets
        if self.find:
            find_result = ActionResult(action_result.action_config)  # type: ignore[arg-type,call-arg]
            self.find.perform(find_result, *object_collections)

            if find_result.is_success and find_result.matches:
                # Extract regions from all matches
                for match in find_result.matches:
                    region = match.get_region()
                    if region is not None:
                        regions.append(region)

        return regions

    def _highlight_regions(
        self,
        regions: list[Region],
        color: tuple[Any, ...],
        thickness: int,
        duration: float,
        flash: bool,
        flash_times: int,
    ) -> bool:
        """Display highlights on the specified regions.

        Args:
            regions: Regions to highlight
            color: RGB color tuple
            thickness: Border thickness in pixels
            duration: Total duration to show highlights
            flash: Whether to flash the highlights
            flash_times: Number of flash cycles

        Returns:
            True if highlighting succeeded
        """
        if flash:
            # Flash mode: alternate visibility
            flash_duration = duration / (flash_times * 2)
            for _i in range(flash_times):
                # Show highlight
                self._draw_highlights(regions, color, thickness)
                time.sleep(flash_duration)

                # Hide highlight
                self._clear_highlights(regions)
                time.sleep(flash_duration)
        else:
            # Static highlight
            self._draw_highlights(regions, color, thickness)
            time.sleep(duration)
            self._clear_highlights(regions)

        return True

    def _draw_highlights(
        self, regions: list[Region], color: tuple[Any, ...], thickness: int
    ) -> None:
        """Draw highlight borders around regions.

        Args:
            regions: Regions to highlight
            color: RGB color tuple
            thickness: Border thickness
        """
        # Placeholder for actual drawing implementation
        # Would interface with platform-specific drawing API
        for _region in regions:
            # Simulate drawing a border around the region
            # In actual implementation, this would draw on screen
            pass

    def _clear_highlights(self, regions: list[Region]) -> None:
        """Clear highlights from regions.

        Args:
            regions: Regions to clear highlights from
        """
        # Placeholder for clearing highlights
        # Would interface with platform-specific drawing API
        for _region in regions:
            # Simulate clearing the highlight
            pass
