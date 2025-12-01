"""Key up action - ported from Qontinui framework.

Release keyboard keys.
"""

import logging
import time
from dataclasses import dataclass

from ....action_interface import ActionInterface
from ....action_result import ActionResult
from ....action_type import ActionType
from ....hal.factory import HALFactory
from ....object_collection import ObjectCollection
from .key_up_options import KeyUpOptions

logger = logging.getLogger(__name__)


@dataclass
class KeyUp(ActionInterface):
    """Releases held keyboard keys.

    Port of KeyUp from Qontinui framework class.

    Low-level keyboard action that releases previously pressed keys.
    Completes keyboard shortcuts and modifier combinations.

    Key features:
    - Multi-key Support: Can release multiple keys in sequence
    - Modifier Handling: Special handling for modifier key release order
    - State Completion: Completes key press operations
    - Timing Control: Configurable pauses between key releases
    """

    def get_action_type(self) -> ActionType:
        """Get action type.

        Returns:
            KEY_UP action type
        """
        return ActionType.KEY_UP

    def perform(
        self, matches: ActionResult, *object_collections: ObjectCollection
    ) -> None:
        """Release keyboard keys.

        Processing behavior:
        - Uses only first ObjectCollection provided
        - Processes multiple StateString objects sequentially
        - Can release modifiers before or after other keys

        Important:
        - Should be paired with previous KeyDown action
        - Modifiers typically released in reverse order of pressing
        - Completes keyboard shortcut sequences

        Args:
            matches: ActionResult with configuration
            object_collections: Collections containing keys to release
        """
        # Get configuration
        if not isinstance(matches.action_config, KeyUpOptions):
            raise ValueError("KeyUp requires KeyUpOptions configuration")

        options = matches.action_config

        # Get HAL controller
        controller = HALFactory.get_input_controller()

        # Optionally release modifiers first
        if options.release_modifiers_first:
            # Release in reverse order (typical for modifiers)
            for modifier in reversed(options.modifiers):
                controller.key_up(modifier.lower())
                logger.debug(f"Key up: {modifier} (modifier)")

        # Process keys from options
        for key in options.keys:
            controller.key_up(key)
            logger.debug(f"Key up: {key}")
            time.sleep(options.pause_between_keys)

        # Process keys from first object collection if present
        if object_collections and object_collections[0].state_strings:
            strings = object_collections[0].state_strings
            for i, state_string in enumerate(strings):
                key = (
                    state_string.string
                    if hasattr(state_string, "string")
                    else str(state_string)
                )
                controller.key_up(key)
                logger.debug(f"Key up: {key}")

                # Pause between keys (except after last one)
                if i < len(strings) - 1:
                    time.sleep(options.pause_between_keys)

        # Release modifiers last if not already released
        if not options.release_modifiers_first:
            for modifier in reversed(options.modifiers):
                controller.key_up(modifier.lower())
                logger.debug(f"Key up: {modifier} (modifier)")

        matches.success = True
