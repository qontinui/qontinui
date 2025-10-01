"""Mouse up action - ported from Qontinui framework.

Release mouse button operations.
"""

import logging
import time
from dataclasses import dataclass

import pyautogui

from ....action_interface import ActionInterface
from ....action_result import ActionResult
from ....action_type import ActionType
from ....object_collection import ObjectCollection
from .mouse_press_options import MouseButton
from .mouse_up_options import MouseUpOptions

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
        press_options = options.get_press_options()

        # Pause before release (using pause_after_press as timing before release)
        if press_options.pause_after_press > 0:
            time.sleep(press_options.pause_after_press)

        # Determine button
        button_map = {
            MouseButton.LEFT: "left",
            MouseButton.RIGHT: "right",
            MouseButton.MIDDLE: "middle",
        }
        button = button_map.get(press_options.button, "left")

        # Release button
        pyautogui.mouseUp(button=button)
        logger.debug(f"Mouse up: {button} button")

        # Pause after release
        if press_options.pause_after_release > 0:
            time.sleep(press_options.pause_after_release)

        matches.success = True
