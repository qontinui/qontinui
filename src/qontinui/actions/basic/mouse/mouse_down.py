"""Mouse down action - ported from Qontinui framework.

Press and hold mouse button operations.
"""

import logging
import time
from dataclasses import dataclass

import pyautogui

from ....action_interface import ActionInterface
from ....action_result import ActionResult
from ....action_type import ActionType
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
        press_options = options.get_press_options()

        # Pause before press (using pause_after_release as timing before press)
        if press_options.pause_after_release > 0:
            time.sleep(press_options.pause_after_release)

        # Determine button
        button_map = {
            MouseButton.LEFT: "left",
            MouseButton.RIGHT: "right",
            MouseButton.MIDDLE: "middle",
        }
        button = button_map.get(press_options.button, "left")

        # Press button
        pyautogui.mouseDown(button=button)
        logger.debug(f"Mouse down: {button} button")

        # Pause after press
        if press_options.pause_after_press > 0:
            time.sleep(press_options.pause_after_press)

        matches.success = True
