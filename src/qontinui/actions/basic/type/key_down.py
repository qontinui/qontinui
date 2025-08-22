"""Key down action - ported from Qontinui framework.

Press and hold keyboard keys.
"""

from dataclasses import dataclass
import time
import logging
import pyautogui
from ....action_interface import ActionInterface
from ....action_type import ActionType
from ....action_result import ActionResult
from ....object_collection import ObjectCollection
from .key_down_options import KeyDownOptions

logger = logging.getLogger(__name__)


@dataclass
class KeyDown(ActionInterface):
    """Presses and holds keyboard keys.
    
    Port of KeyDown from Qontinui framework class.
    
    Low-level keyboard action that initiates key presses without releasing.
    Essential for keyboard shortcuts, modifier combinations, and sustained presses.
    
    Key features:
    - Multi-key Support: Can press multiple keys in sequence
    - Modifier Integration: Special handling for modifier keys
    - State Persistence: Keys remain pressed until KeyUp
    - Timing Control: Configurable pauses between key presses
    """
    
    def get_action_type(self) -> ActionType:
        """Get action type.
        
        Returns:
            KEY_DOWN action type
        """
        return ActionType.KEY_DOWN
    
    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Press and hold keyboard keys.
        
        Processing behavior:
        - Uses only first ObjectCollection provided
        - Processes multiple StateString objects sequentially
        - Modifier keys from options applied to each key
        
        Important:
        - Always pair with KeyUp to avoid leaving keys pressed
        - Modifier keys should be released in reverse order
        - Platform-specific key handling may affect combinations
        
        Args:
            matches: ActionResult with configuration
            object_collections: Collections containing keys to press
        """
        # Get configuration
        if not isinstance(matches.action_config, KeyDownOptions):
            raise ValueError("KeyDown requires KeyDownOptions configuration")
        
        options = matches.action_config
        
        # Apply modifiers first
        for modifier in options.modifiers:
            pyautogui.keyDown(modifier.lower())
            logger.debug(f"Key down: {modifier} (modifier)")
        
        # Process keys from options
        for key in options.keys:
            pyautogui.keyDown(key)
            logger.debug(f"Key down: {key}")
            time.sleep(options.pause_between_keys)
        
        # Process keys from first object collection if present
        if object_collections and object_collections[0].state_strings:
            strings = object_collections[0].state_strings
            for i, state_string in enumerate(strings):
                key = state_string.string if hasattr(state_string, 'string') else str(state_string)
                pyautogui.keyDown(key)
                logger.debug(f"Key down: {key}")
                
                # Pause between keys (except after last one)
                if i < len(strings) - 1:
                    time.sleep(options.pause_between_keys)
        
        matches.success = True