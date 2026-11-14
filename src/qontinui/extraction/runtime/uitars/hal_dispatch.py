"""Dispatch UI-TARS actions through the HAL input controllers.

UI-TARS emits abstract actions (click, type, scroll, drag, ...). This module
turns one such action into calls on the HAL mouse/keyboard controllers
(pynput by default), so UI-TARS shares the single input stack the rest of
qontinui uses.

UI-TARS was tuned against pyautogui, which gave every call two behaviours that
pynput does not have. Both are reproduced here explicitly:

* **Corner abort (failsafe).** pyautogui 0.9.54 raised ``FailSafeException``
  when the pointer sat on any of the four corners ``(0, 0)``, ``(0, h-1)``,
  ``(w-1, 0)``, ``(w-1, h-1)`` of the rectangle its ``size()`` reported. That
  rectangle differs by platform: on X11 it is the whole X screen (all
  monitors, ``width_in_pixels`` x ``height_in_pixels``); on Windows it is the
  primary monitor (``SM_CXSCREEN`` x ``SM_CYSCREEN``), whose origin is
  ``(0, 0)``; on macOS the main display, also at ``(0, 0)``. Before every
  input step the pointer position is read and, on a corner, the action is
  refused with a logged error. Moving the mouse into a corner is how an
  operator stops an autonomous run. The rectangle is derived from the HAL
  screen capture's monitors by :func:`failsafe_rect_from_monitors`; when it
  cannot be determined, only ``(0, 0)`` is checked and that degradation is
  logged once.
* **Pause after every input step.** pyautogui slept ``PAUSE`` (0.1 s) after
  every call. ``STEP_PAUSE_S`` is slept after each input step, so a compound
  action (click then type, move then drag) waits between its steps and every
  input action leaves the UI 0.1 s to react before the caller looks at the
  screen again.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Sequence

from qontinui.hal.interfaces.input_controller import MouseButton
from qontinui.hal.interfaces.keyboard_controller import IKeyboardController
from qontinui.hal.interfaces.mouse_controller import IMouseController
from qontinui.hal.interfaces.screen_capture import Monitor

logger = logging.getLogger(__name__)

#: Pause after every input step (pyautogui's ``PAUSE``), in seconds.
STEP_PAUSE_S = 0.1

#: Typing interval per character (seconds).
TYPE_INTERVAL_S = 0.02

#: Scroll amount used when UI-TARS gives none.
DEFAULT_SCROLL_AMOUNT = 100

#: Windows wheel delta of one notch (``WHEEL_DELTA`` in winuser.h).
WIN32_WHEEL_DELTA = 120

_warned_no_failsafe_rect = False

#: ``(left, top, width, height)`` of the failsafe rectangle.
Rect = tuple[int, int, int, int]


class FailsafeTriggered(RuntimeError):
    """Raised internally when the pointer sits on a failsafe corner."""


def _current_platform() -> str:
    """Return ``sys.platform`` (a seam for tests)."""
    return sys.platform


def scroll_amount_to_clicks(amount: int, platform: str | None = None) -> int:
    """Convert a UI-TARS scroll amount into HAL ``mouse_scroll`` clicks.

    UI-TARS amounts were tuned against pyautogui, whose unit differs by
    platform:

    * **X11 (linux):** pyautogui sent ``amount`` wheel-button clicks, which is
      exactly what pynput's ``scroll`` does, so the amount passes through.
    * **Windows:** pyautogui passed ``amount`` as the raw wheel delta (120 =
      one notch), while pynput multiplies by ``WHEEL_DELTA`` itself; passing
      the amount through would scroll 120x further. The amount is converted to
      notches, keeping its sign and scrolling at least one notch.
    * **macOS: approximate, unverified.** pyautogui posted ``amount`` *line*
      units (``kCGScrollEventUnitLine``); pynput posts ``amount * 10`` *pixel*
      units (``kCGScrollEventUnitPixel``, ``_SCROLL_SPEED = 10``). How many
      pixels a line is depends on the event source and the receiving
      application, so no exact conversion exists; the amount passes through
      and the result matches pyautogui only where a line is about 10 pixels.

    Args:
        amount: Signed scroll amount in pyautogui units
        platform: ``sys.platform`` value to convert for (defaults to the
            running platform)

    Returns:
        Signed number of clicks for ``IMouseController.mouse_scroll``
    """
    platform = _current_platform() if platform is None else platform
    if platform != "win32" or amount == 0:
        return amount
    notches = max(1, round(abs(amount) / WIN32_WHEEL_DELTA))
    return notches if amount > 0 else -notches


def failsafe_rect_from_monitors(monitors: Sequence[Monitor], platform: str) -> Rect | None:
    """Return the rectangle whose corners pyautogui treated as failsafe points.

    One rule per platform family, matching what pyautogui's ``size()`` read:

    * **linux (X11):** the whole X screen, i.e. the bounding box of every
      monitor. pyautogui used ``Display.screen().width_in_pixels`` /
      ``height_in_pixels``, which on a multi-monitor X screen spans all of
      them.
    * **everything else (Windows, macOS):** the monitor whose origin is
      ``(0, 0)``. Windows' ``SM_CXSCREEN``/``SM_CYSCREEN`` and macOS's main
      display both describe that monitor; the order mss lists monitors in is
      not used, because it does not reliably put the primary first.

    Args:
        monitors: Physical monitors (``IScreenCapture.get_monitors()``)
        platform: ``sys.platform`` value

    Returns:
        ``(left, top, width, height)``, or None when no rectangle can be
        determined (no monitors, or no monitor at the origin)
    """
    if not monitors:
        return None
    if platform.startswith("linux"):
        left = min(m.x for m in monitors)
        top = min(m.y for m in monitors)
        right = max(m.x + m.width for m in monitors)
        bottom = max(m.y + m.height for m in monitors)
        return (left, top, right - left, bottom - top)
    for m in monitors:
        if (m.x, m.y) == (0, 0):
            return (0, 0, m.width, m.height)
    return None


def failsafe_corners(rect: Rect | None) -> frozenset[tuple[int, int]]:
    """Return the four corner pixels of ``rect`` (only ``(0, 0)`` when None)."""
    if rect is None:
        return frozenset({(0, 0)})
    left, top, width, height = rect
    right, bottom = left + width - 1, top + height - 1
    return frozenset({(left, top), (left, bottom), (right, top), (right, bottom)})


def _resolve_failsafe_rect() -> Rect | None:
    """Derive the failsafe rectangle from the HAL screen capture, or None."""
    global _warned_no_failsafe_rect
    reason: str
    try:
        from qontinui.hal.factory import HALFactory

        monitors = HALFactory.get_screen_capture().get_monitors()
        rect = failsafe_rect_from_monitors(monitors, _current_platform())
        if rect is not None:
            return rect
        reason = f"no monitor at the origin among {len(monitors)}"
    except Exception as e:
        reason = str(e)
    if not _warned_no_failsafe_rect:
        _warned_no_failsafe_rect = True
        logger.warning(
            f"Failsafe screen rectangle unavailable ({reason}); the UI-TARS "
            "failsafe checks only the (0, 0) corner"
        )
    return None


def _check_failsafe(mouse: IMouseController, corners: frozenset[tuple[int, int]]) -> None:
    """Raise FailsafeTriggered if the pointer is on a failsafe corner."""
    pos = mouse.get_mouse_position()
    if (pos.x, pos.y) in corners:
        raise FailsafeTriggered(
            f"pointer is at the failsafe corner {(pos.x, pos.y)}; "
            "refusing to execute UI-TARS action"
        )


async def _pause_after_step() -> None:
    """Wait after an input step, as pyautogui's ``PAUSE`` did."""
    await asyncio.sleep(STEP_PAUSE_S)


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
    failsafe_rect: Rect | None = None,
    incomplete_action_result: bool = True,
) -> bool:
    """Execute one UI-TARS action through the HAL controllers.

    Args:
        action_type: Action type value (``"click"``, ``"type"``, ``"scroll"``, ...)
        x, y: Target coordinates for pointer actions
        text: Text for a type action
        scroll_direction: ``"up"``, ``"down"``, ``"left"`` or ``"right"``
        scroll_amount: Scroll amount in pyautogui units (defaults to
            ``DEFAULT_SCROLL_AMOUNT``; see :func:`scroll_amount_to_clicks`)
        keys: Keys for a hotkey action (pyautogui key names are accepted)
        duration: Seconds for a wait action
        end_x, end_y: Drag end coordinates
        mouse: Mouse controller (defaults to ``HALFactory.get_mouse_controller()``)
        keyboard: Keyboard controller (defaults to
            ``HALFactory.get_keyboard_controller()``)
        failsafe_rect: ``(left, top, width, height)`` whose corners abort the
            action (defaults to :func:`failsafe_rect_from_monitors` over the
            HAL screen capture's monitors)
        incomplete_action_result: What to return for a pointer action without
            the coordinates it needs, or a hotkey without keys. Nothing is
            executed either way; callers keep their historical answer.

    Returns:
        True if the action was executed and every controller call reported
        success; False if it was unknown, refused by the failsafe, or failed.
    """
    if action_type == "wait":
        await asyncio.sleep(duration or 1.0)
        return True
    if action_type == "done":
        return True

    try:
        mouse, keyboard = _resolve_controllers(mouse, keyboard)
        corners = failsafe_corners(
            failsafe_rect if failsafe_rect is not None else _resolve_failsafe_rect()
        )

        async def step(call: object) -> bool:
            """Pause after an executed step and report its result."""
            await _pause_after_step()
            return bool(call)

        def guard() -> None:
            _check_failsafe(mouse, corners)

        if action_type == "click" and x is not None and y is not None:
            guard()
            return await step(mouse.click_at(x, y))
        if action_type == "double_click" and x is not None and y is not None:
            guard()
            return await step(mouse.double_click_at(x, y))
        if action_type == "right_click" and x is not None and y is not None:
            guard()
            return await step(mouse.click_at(x, y, button=MouseButton.RIGHT))
        if action_type == "type":
            if x is not None and y is not None:
                guard()
                if not await step(mouse.click_at(x, y)):
                    return False
            if text:
                guard()
                return await step(keyboard.type_text(text, interval=TYPE_INTERVAL_S))
            return True
        if action_type == "scroll":
            amount = scroll_amount_to_clicks(scroll_amount or DEFAULT_SCROLL_AMOUNT)
            axes = {
                "up": (amount, 0),
                "down": (-amount, 0),
                "left": (0, -amount),
                "right": (0, amount),
            }.get(scroll_direction or "")
            if axes is None:
                # Historical behaviour: an unknown direction is a no-op success.
                logger.warning(f"Unknown scroll direction: {scroll_direction}")
                return True
            guard()
            return await step(mouse.mouse_scroll(axes[0], horizontal_clicks=axes[1]))
        if action_type == "hover" and x is not None and y is not None:
            guard()
            return await step(mouse.move_mouse(x, y))
        if (
            action_type == "drag"
            and x is not None
            and y is not None
            and end_x is not None
            and end_y is not None
        ):
            guard()
            if not await step(mouse.move_mouse(x, y)):
                return False
            guard()
            return await step(mouse.drag(x, y, end_x, end_y))
        if action_type == "hotkey" and keys:
            guard()
            return await step(keyboard.hotkey(*keys))
        if action_type in ("click", "double_click", "right_click", "hover", "drag"):
            logger.debug(
                f"UI-TARS {action_type} action without coordinates "
                f"(x={x}, y={y}, end_x={end_x}, end_y={end_y}); skipped"
            )
            return incomplete_action_result
        if action_type == "hotkey":
            logger.debug("UI-TARS hotkey action without keys; skipped")
            return incomplete_action_result

        logger.warning(f"Unknown action type: {action_type}")
        return False

    except FailsafeTriggered as e:
        logger.error(f"UI-TARS action '{action_type}' aborted: {e}")
        return False
    except Exception as e:
        logger.error(f"HAL action '{action_type}' failed: {e}")
        return False
