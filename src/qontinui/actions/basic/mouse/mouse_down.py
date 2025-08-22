"""Mouse down action - ported from Qontinui framework.

Press and hold mouse button operations.
"""

from dataclasses import dataclass
import time
import logging
import pyautogui
from ....action_interface import ActionInterface
from ....action_type import ActionType
from ....action_result import ActionResult
from ....object_collection import ObjectCollection
from .mouse_down_options import MouseDownOptions
from .mouse_press_options import MouseButton

logger = logging.getLogger(__name__)


@dataclass
class MouseDown(ActionInterface):
    """Presses and holds a mouse button.
    
    Port of MouseDown from Qontinui framework class.
    
    Low-level action that initiates a mouse button press without releasing.
    Essential for drag-and-drop operations and sustained mouse interactions.
    
    Key features:
    - Button Control: Supports left, right, and middle mouse buttons
    - Timing Precision: Configurable pauses before and after press
    - State Persistence: Maintains button state until MouseUp
    - Platform Independence: Works consistently across OSes
    """
    
    def get_action_type(self) -> ActionType:
        """Get action type.
        
        Returns:
            MOUSE_DOWN action type
        """
        return ActionType.MOUSE_DOWN
    
    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Press and hold mouse button.
        
        Important:
        - Always pair with MouseUp to avoid leaving mouse pressed
        - Mouse position is not changed by this action
        - Some applications have timeouts for sustained presses
        
        Args:
            matches: ActionResult with configuration
            object_collections: Not used for this action
        """
        # Get configuration
        if not isinstance(matches.action_config, MouseDownOptions):
            raise ValueError("MouseDown requires MouseDownOptions configuration")
        
        options = matches.action_config
        
        # Pause before press
        if options.pause_before_mouse_down > 0:
            time.sleep(options.pause_before_mouse_down)
        
        # Determine button
        button_map = {
            MouseButton.LEFT: 'left',
            MouseButton.RIGHT: 'right',
            MouseButton.MIDDLE: 'middle'
        }
        button = button_map.get(options.button, 'left')
        
        # Press button
        pyautogui.mouseDown(button=button)
        logger.debug(f"Mouse down: {button} button")
        
        # Pause after press
        if options.pause_after_mouse_down > 0:
            time.sleep(options.pause_after_mouse_down)
        
        matches.success = True