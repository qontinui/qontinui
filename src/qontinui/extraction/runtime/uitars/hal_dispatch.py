"""Dispatch UI-TARS actions through the HAL input controllers.

UI-TARS emits abstract actions (click, type, scroll, drag, ...). This module
turns one such action into calls on the HAL mouse/keyboard controllers
(pynput by default), so UI-TARS shares the single input stack the rest of
qontinui uses.

Two behaviours UI-TARS runs rely on, which pynput does not provide by
itself, are implemented explicitly here:

* **Corner abort (failsafe).** Before every input step the pointer position
  is read; if it sits at the top-left corner ``(0, 0)`` the action is refused
  with a logged error. Moving the mouse into that corner is how an operator
  stops an autonomous run.
* **Pause between compound steps.** A compound action (click then type,
  move then drag) waits ``COMPOUND_STEP_PAUSE_S`` between its steps so the
  target UI can react to the first step before the second arrives.
"""

from __future__ import annotations

import asyncio
import logging

from qontinui.hal.interfaces.input_controller import MouseButton
from qontinui.hal.interfaces.keyboard_controller import IKeyboardController
from qontinui.hal.interfaces.mouse_controller import IMouseController

logger = logging.getLogger(__name__)

#: Pause between the steps of a compound action (seconds).
COMPOUND_STEP_PAUSE_S = 0.1

#: Pointer position that aborts the action (operator failsafe).
FAILSAFE_CORNER = (0, 0)

#: Typing interval per character (seconds).
TYPE_INTERVAL_S = 0.02

#: Scroll amount used when UI-TARS gives none.
DEFAULT_SCROLL_AMOUNT = 100


class FailsafeTriggered(RuntimeError):
    """Raised internally when the pointer sits in the failsafe corner."""


def _check_failsafe(mouse: IMouseController) -> None:
    """Raise FailsafeTriggered if the pointer is in the failsafe corner."""
    pos = mouse.get_mouse_position()
    if (pos.x, pos.y) == FAILSAFE_CORNER:
        raise FailsafeTriggered(
            f"pointer is at the failsafe corner {FAILSAFE_CORNER}; "
            "refusing to execute UI-TARS action"
        )


async def _pause_between_steps() -> None:
    """Wait between the steps of a compound action."""
    await asyncio.sleep(COMPOUND_STEP_PAUSE_S)


def _resolve_controllers(
    mouse: IMouseController | None, keyboard: IKeyboardController | None
) -> tuple[IMouseController, IKeyboardController]:
    """Return the given controllers, filling gaps from the HAL factory."""
    if mouse is None or keyboard is None:
        from qontinui.hal.factory import HALFactory

        if mouse is None:
            mouse = HALFactory.get_mouse_controller()
        if keyboard is None:
            keyboard = HALFactory.get_keyboard_controller()
    return mouse, keyboard


async def execute_hal_action(
    action_type: str,
    *,
    x: int | None = None,
    y: int | None = None,
    text: str | None = None,
    scroll_direction: str | None = None,
    scroll_amount: int | None = None,
    keys: list[str] | None = None,
    duration: float | None = None,
    end_x: int | None = None,
    end_y: int | None = None,
    mouse: IMouseController | None = None,
    keyboard: IKeyboardController | None = None,
) -> bool:
    """Execute one UI-TARS action through the HAL controllers.

    Args:
        action_type: Action type value (``"click"``, ``"type"``, ``"scroll"``, ...)
        x, y: Target coordinates for pointer actions
        text: Text for a type action
        scroll_direction: ``"up"``, ``"down"``, ``"left"`` or ``"right"``
        scroll_amount: Scroll clicks (defaults to ``DEFAULT_SCROLL_AMOUNT``)
        keys: Keys for a hotkey action
        duration: Seconds for a wait action
        end_x, end_y: Drag end coordinates
        mouse: Mouse controller (defaults to ``HALFactory.get_mouse_controller()``)
        keyboard: Keyboard controller (defaults to
            ``HALFactory.get_keyboard_controller()``)

    Returns:
        True if the action was executed, False if it was unknown, refused by
        the failsafe, or failed.
    """
    if action_type == "wait":
        await asyncio.sleep(duration or 1.0)
        return True
    if action_type == "done":
        return True

    try:
        mouse, keyboard = _resolve_controllers(mouse, keyboard)

        if action_type == "click" and x is not None and y is not None:
            _check_failsafe(mouse)
            mouse.click_at(x, y)
        elif action_type == "double_click" and x is not None and y is not None:
            _check_failsafe(mouse)
            mouse.double_click_at(x, y)
        elif action_type == "right_click" and x is not None and y is not None:
            _check_failsafe(mouse)
            mouse.click_at(x, y, button=MouseButton.RIGHT)
        elif action_type == "type":
            if x is not None and y is not None:
                _check_failsafe(mouse)
                mouse.click_at(x, y)
                if text:
                    await _pause_between_steps()
            if text:
                _check_failsafe(mouse)
                keyboard.type_text(text, interval=TYPE_INTERVAL_S)
        elif action_type == "scroll":
            amount = scroll_amount or DEFAULT_SCROLL_AMOUNT
            vertical, horizontal = {
                "up": (amount, 0),
                "down": (-amount, 0),
                "left": (0, -amount),
                "right": (0, amount),
            }.get(scroll_direction or "", (0, 0))
            if (vertical, horizontal) == (0, 0):
                logger.warning(f"Unknown scroll direction: {scroll_direction}")
                return False
            _check_failsafe(mouse)
            mouse.mouse_scroll(vertical, horizontal_clicks=horizontal)
        elif action_type == "hover" and x is not None and y is not None:
            _check_failsafe(mouse)
            mouse.move_mouse(x, y)
        elif (
            action_type == "drag"
            and x is not None
            and y is not None
            and end_x is not None
            and end_y is not None
        ):
            _check_failsafe(mouse)
            mouse.move_mouse(x, y)
            await _pause_between_steps()
            _check_failsafe(mouse)
            mouse.drag(x, y, end_x, end_y)
        elif action_type == "hotkey" and keys:
            _check_failsafe(mouse)
            keyboard.hotkey(*keys)
        elif action_type in ("click", "double_click", "right_click", "hover", "drag"):
            # Pointer action without the coordinates it needs: nothing to do.
            logger.debug(
                f"UI-TARS {action_type} action without coordinates "
                f"(x={x}, y={y}, end_x={end_x}, end_y={end_y}); skipped"
            )
        elif action_type == "hotkey":
            logger.debug("UI-TARS hotkey action without keys; skipped")
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return False

        return True

    except FailsafeTriggered as e:
        logger.error(f"UI-TARS action '{action_type}' aborted: {e}")
        return False
    except Exception as e:
        logger.error(f"HAL action '{action_type}' failed: {e}")
        return False
