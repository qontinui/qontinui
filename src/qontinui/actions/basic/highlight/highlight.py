"""Highlight action implementation - ported from Qontinui framework.

Highlights regions on screen for debugging and visualization.
"""

import time
from typing import Optional, List
from ...action_interface import ActionInterface
from ...action_result import ActionResult
from ...object_collection import ObjectCollection
from .highlight_options import HighlightOptions
from ..find.find import Find
from ....model.region import Region


class Highlight(ActionInterface):
    """Highlights regions on screen for debugging and visualization.
    
    Port of Highlight from Qontinui framework action.
    
    The Highlight action is primarily used for:
    - Debugging pattern matching results
    - Providing visual feedback during automation
    - Demonstrating where actions will occur
    """
    
    def __init__(self, find: Optional[Find] = None):
        """Initialize Highlight action.
        
        Args:
            find: The Find action for locating regions to highlight
        """
        self.find = find
    
    def perform(self, action_result: ActionResult, *object_collections: ObjectCollection) -> None:
        """Execute the highlight operation.
        
        Finds target regions and displays visual highlights around them.
        
        Args:
            action_result: Result container that will be populated
            object_collections: Collections defining what to highlight
        """
        if not isinstance(action_result.action_config, HighlightOptions):
            action_result.set_success(False)
            return
        
        highlight_options = action_result.action_config
        
        # Find regions to highlight
        regions = self._find_regions_to_highlight(action_result, object_collections)
        if not regions:
            action_result.set_success(False)
            action_result.text = "No regions found to highlight"
            return
        
        # Perform highlighting
        success = self._highlight_regions(
            regions,
            highlight_options.get_color(),
            highlight_options.get_thickness(),
            highlight_options.get_highlight_duration(),
            highlight_options.is_flash(),
            highlight_options.get_flash_times()
        )
        
        action_result.set_success(success)
        if success:
            action_result.text = f"Highlighted {len(regions)} region(s)"
    
    def _find_regions_to_highlight(self,
                                  action_result: ActionResult,
                                  object_collections: tuple) -> List[Region]:
        """Find regions to highlight.
        
        Args:
            action_result: The action result for storing find results
            object_collections: Collections to search in
            
        Returns:
            List of regions to highlight
        """
        regions = []
        
        if not object_collections:
            return regions
        
        # Use Find to locate targets
        if self.find:
            find_result = ActionResult(action_result.action_config)
            self.find.perform(find_result, *object_collections)
            
            if find_result.is_success() and find_result.match_list:
                # Extract regions from all matches
                for match in find_result.match_list:
                    regions.append(match.get_region())
        
        return regions
    
    def _highlight_regions(self,
                          regions: List[Region],
                          color: tuple,
                          thickness: int,
                          duration: float,
                          flash: bool,
                          flash_times: int) -> bool:
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
            for i in range(flash_times):
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
    
    def _draw_highlights(self,
                        regions: List[Region],
                        color: tuple,
                        thickness: int) -> None:
        """Draw highlight borders around regions.
        
        Args:
            regions: Regions to highlight
            color: RGB color tuple
            thickness: Border thickness
        """
        # Placeholder for actual drawing implementation
        # Would interface with platform-specific drawing API
        for region in regions:
            # Simulate drawing a border around the region
            # In actual implementation, this would draw on screen
            pass
    
    def _clear_highlights(self, regions: List[Region]) -> None:
        """Clear highlights from regions.
        
        Args:
            regions: Regions to clear highlights from
        """
        # Placeholder for clearing highlights
        # Would interface with platform-specific drawing API
        for region in regions:
            # Simulate clearing the highlight
            pass