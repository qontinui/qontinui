"""Mouse up action - ported from Qontinui framework.

Release mouse button operations.
"""

from dataclasses import dataclass
import time
import logging
import pyautogui
from ....action_interface import ActionInterface
from ....action_type import ActionType
from ....action_result import ActionResult
from ....object_collection import ObjectCollection
from .mouse_up_options import MouseUpOptions
from .mouse_press_options import MouseButton

logger = logging.getLogger(__name__)


@dataclass
class MouseUp(ActionInterface):
    """Releases a held mouse button.
    
    Port of MouseUp from Qontinui framework class.
    
    Low-level action that releases a previously pressed mouse button.
    Completes drag-and-drop operations and other sustained interactions.
    
    Key features:
    - Button Control: Releases left, right, or middle mouse button
    - Timing Precision: Configurable pauses before and after release
    - State Completion: Completes mouse press operations
    - Platform Independence: Works consistently across OSes
    """
    
    def get_action_type(self) -> ActionType:
        """Get action type.
        
        Returns:
            MOUSE_UP action type
        """
        return ActionType.MOUSE_UP
    
    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Release mouse button.
        
        Important:
        - Should be paired with a previous MouseDown action
        - Mouse position is not changed by this action
        - Completes drag operations when preceded by MouseDown and MouseMove
        
        Args:
            matches: ActionResult with configuration
            object_collections: Not used for this action
        """
        # Get configuration
        if not isinstance(matches.action_config, MouseUpOptions):
            raise ValueError("MouseUp requires MouseUpOptions configuration")
        
        options = matches.action_config
        
        # Pause before release
        if options.pause_before_mouse_up > 0:
            time.sleep(options.pause_before_mouse_up)
        
        # Determine button
        button_map = {
            MouseButton.LEFT: 'left',
            MouseButton.RIGHT: 'right',
            MouseButton.MIDDLE: 'middle'
        }
        button = button_map.get(options.button, 'left')
        
        # Release button
        pyautogui.mouseUp(button=button)
        logger.debug(f"Mouse up: {button} button")
        
        # Pause after release
        if options.pause_after_mouse_up > 0:
            time.sleep(options.pause_after_mouse_up)
        
        matches.success = True