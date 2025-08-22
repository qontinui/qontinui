"""Wait vanish action - ported from Qontinui framework.

Waits for visual elements to disappear from screen.
"""

from dataclasses import dataclass
import time
import logging
from ....action_interface import ActionInterface
from ....action_result import ActionResult
from ....object_collection import ObjectCollection
from ..find.find import Find
from ..vanish.vanish_options import VanishOptions

logger = logging.getLogger(__name__)


@dataclass
class WaitVanish(ActionInterface):
    """Waits for visual elements to disappear from screen.
    
    Port of WaitVanish from Qontinui framework class.
    
    Monitors GUI for absence of specified elements.
    Critical for synchronization and state transition detection.
    """
    
    find: Find
    
    def get_action_type(self) -> str:
        """Get action type.
        
        Returns:
            VANISH action type
        """
        return "VANISH"
    
    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Perform vanish wait.
        
        Args:
            matches: Action result to populate
            object_collections: Objects to wait for vanishing
        """
        # Get configuration
        config = matches.action_config
        timeout = 10.0  # Default timeout
        
        if isinstance(config, VanishOptions):
            timeout = config.timeout
        
        # Process only first collection (as per Brobot)
        if not object_collections:
            return
        
        first_collection = object_collections[0]
        
        # Keep checking until objects vanish or timeout
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            # Clear previous matches for fresh search
            matches.clear_matches()
            
            # Search for objects
            self.find.perform(matches, first_collection)
            
            # If nothing found, objects have vanished - success!
            if not matches.match_locations:
                matches.success = True
                logger.debug("Objects vanished successfully")
                break
            
            # Check if we should continue
            if not self._is_ok_to_continue(matches, len(first_collection.state_images)):
                logger.debug("Stopping vanish wait due to action lifecycle")
                break
            
            # Brief pause before next check
            time.sleep(0.1)
        
        if not matches.success:
            logger.debug(f"Vanish timeout after {timeout} seconds")
    
    def _is_ok_to_continue(self, matches: ActionResult, num_images: int) -> bool:
        """Check if action should continue.
        
        This is a simplified version of Brobot's ActionLifecycleManagement.
        
        Args:
            matches: Current matches
            num_images: Number of images being searched
            
        Returns:
            True if should continue
        """
        # Simple implementation - could be enhanced with more sophisticated checks
        # For now, always continue unless explicitly stopped
        return True