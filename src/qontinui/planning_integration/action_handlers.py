"""Action handlers bridging HTN plan actions to HAL and UI Bridge.

Each handler follows the signature: handler(blackboard: Blackboard, *args) -> None
Handlers raise RuntimeError on failure (PlanExecutor catches and triggers replanning).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from qontinui.discovery.target_connection import Element

logger = logging.getLogger(__name__)

# Polling defaults for wait_* handlers
_MAX_POLL_RETRIES: int = 10
_POLL_INTERVAL_S: float = 0.5


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from synchronous code.

    Handles the case where an event loop may or may not already be running.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We're inside a running loop (e.g. Jupyter, nested async).
        # Create a new thread to run the coroutine.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


async def _find_element_async(ui_connection: Any, element_id: str) -> Element:
    """Query the UI Bridge connection for an element by ID.

    Returns the Element if found; raises RuntimeError otherwise.
    """
    elements: list[Element] = await ui_connection.find_elements()
    for elem in elements:
        if elem.id == element_id:
            return elem
    raise RuntimeError(f"Element '{element_id}' not found via UI Bridge")


def _find_element(blackboard: Any, element_id: str) -> Element:
    """Synchronous wrapper to find an element via the UI Bridge connection.

    Looks up ``_ui_connection`` from the blackboard, queries it, and
    returns the matching Element or raises RuntimeError.
    """
    ui_connection = blackboard.get("_ui_connection")
    if ui_connection is None:
        raise RuntimeError(
            "No UI Bridge connection available in blackboard (_ui_connection)"
        )
    return _run_async(_find_element_async(ui_connection, element_id))


def _refresh_elements(blackboard: Any) -> list[Element]:
    """Re-fetch all elements and cache them on the blackboard."""
    ui_connection = blackboard.get("_ui_connection")
    if ui_connection is None:
        return []
    elements: list[Element] = _run_async(ui_connection.find_elements())
    blackboard.set("_ui_elements", elements)
    return elements


def create_action_handlers(
    hal: Any,
    ui_connection: Any | None = None,
    state_manager: Any | None = None,
) -> dict[str, Callable[..., None]]:
    """Build a dict of action handlers for the HTN PlanExecutor.

    Args:
        hal: An initialized HALContainer (from ``qontinui.hal.initialize_hal()``).
        ui_connection: Optional UI Bridge connection (async context manager).
            If provided, it is also stored in the blackboard by the caller
            via ``blackboard.set("_ui_connection", conn)``.
        state_manager: Optional multistate StateManager for navigation actions.

    Returns:
        Dictionary mapping action names to handler callables.  Each handler
        has the signature ``handler(blackboard, *args) -> None`` and raises
        ``RuntimeError`` on failure.
    """

    # ------------------------------------------------------------------
    # click_element(blackboard, element_id)
    # ------------------------------------------------------------------
    def click_element(blackboard: Any, element_id: str) -> None:
        """Click an element identified by its UI Bridge element ID."""
        logger.debug("click_element: %s", element_id)
        element = _find_element(blackboard, element_id)

        if element.bbox is None:
            raise RuntimeError(
                f"Element '{element_id}' has no bounding box; cannot click"
            )

        cx, cy = element.bbox.center
        hal.mouse_controller.mouse_click(int(cx), int(cy))
        logger.info("Clicked element '%s' at (%d, %d)", element_id, int(cx), int(cy))

        # Refresh element cache (the click may have changed the DOM)
        _refresh_elements(blackboard)

    # ------------------------------------------------------------------
    # type_text(blackboard, element_id, text)
    # ------------------------------------------------------------------
    def type_text(blackboard: Any, element_id: str, text: str) -> None:
        """Type text, optionally targeting a specific element first.

        If *element_id* is ``"_keyboard"``, text is typed directly without
        focusing any element (useful for keyboard shortcuts / escape).
        Otherwise the element is clicked first to ensure focus.
        """
        logger.debug("type_text: element=%s text=%r", element_id, text)

        if element_id != "_keyboard":
            # Click the element to focus it before typing
            click_element(blackboard, element_id)

        hal.keyboard_controller.type_text(text)
        logger.info("Typed text into '%s'", element_id)

    # ------------------------------------------------------------------
    # navigate_path(blackboard, target_state)
    # ------------------------------------------------------------------
    def navigate_path(blackboard: Any, target_state: str) -> None:
        """Navigate to *target_state* using the state manager."""
        mgr = blackboard.get("_state_manager") or state_manager
        if mgr is None:
            raise RuntimeError(
                "No state manager available for navigate_path"
            )
        logger.debug("navigate_path: %s", target_state)
        success = mgr.navigate_to([target_state])
        if not success:
            raise RuntimeError(f"navigate_path failed: could not reach '{target_state}'")
        logger.info("Navigated to state '%s'", target_state)

    # ------------------------------------------------------------------
    # navigate_transition(blackboard, transition_id)
    # ------------------------------------------------------------------
    def navigate_transition(blackboard: Any, transition_id: str) -> None:
        """Execute a specific transition by ID."""
        mgr = blackboard.get("_state_manager") or state_manager
        if mgr is None:
            raise RuntimeError(
                "No state manager available for navigate_transition"
            )
        logger.debug("navigate_transition: %s", transition_id)
        success = mgr.execute_transition(transition_id)
        if not success:
            raise RuntimeError(
                f"navigate_transition failed: transition '{transition_id}' did not succeed"
            )
        logger.info("Executed transition '%s'", transition_id)

    # ------------------------------------------------------------------
    # wait_for_state(blackboard, target_state)
    # ------------------------------------------------------------------
    def wait_for_state(blackboard: Any, target_state: str) -> None:
        """Poll until *target_state* appears in the active state set.

        Retries up to ``_MAX_POLL_RETRIES`` times with ``_POLL_INTERVAL_S``
        delay.  Raises RuntimeError on timeout.
        """
        mgr = blackboard.get("_state_manager") or state_manager
        if mgr is None:
            raise RuntimeError("No state manager available for wait_for_state")

        logger.debug("wait_for_state: %s", target_state)
        for attempt in range(_MAX_POLL_RETRIES):
            active = mgr.get_active_states()
            if target_state in active:
                logger.info(
                    "State '%s' became active (attempt %d)", target_state, attempt + 1
                )
                return
            time.sleep(_POLL_INTERVAL_S)

        raise RuntimeError(
            f"Timed out waiting for state '{target_state}' after "
            f"{_MAX_POLL_RETRIES * _POLL_INTERVAL_S:.1f}s"
        )

    # ------------------------------------------------------------------
    # wait_for_element(blackboard, element_id)
    # ------------------------------------------------------------------
    def wait_for_element(blackboard: Any, element_id: str) -> None:
        """Poll until *element_id* becomes visible via UI Bridge.

        Retries up to ``_MAX_POLL_RETRIES`` times with ``_POLL_INTERVAL_S``
        delay.  Raises RuntimeError on timeout.
        """
        ui_conn = blackboard.get("_ui_connection")
        if ui_conn is None:
            raise RuntimeError(
                "No UI Bridge connection available for wait_for_element"
            )

        logger.debug("wait_for_element: %s", element_id)
        for attempt in range(_MAX_POLL_RETRIES):
            elements: list[Element] = _run_async(ui_conn.find_elements())
            for elem in elements:
                if elem.id == element_id and elem.is_visible:
                    logger.info(
                        "Element '%s' became visible (attempt %d)",
                        element_id,
                        attempt + 1,
                    )
                    blackboard.set("_ui_elements", elements)
                    return
            time.sleep(_POLL_INTERVAL_S)

        raise RuntimeError(
            f"Timed out waiting for element '{element_id}' after "
            f"{_MAX_POLL_RETRIES * _POLL_INTERVAL_S:.1f}s"
        )

    # ------------------------------------------------------------------
    # dismiss_dialog(blackboard, dialog_state)
    # ------------------------------------------------------------------
    def dismiss_dialog(blackboard: Any, dialog_state: str) -> None:
        """Dismiss a dialog by pressing Escape.

        After pressing Escape, waits briefly and verifies the dialog state
        has been deactivated (if a state manager is available).
        """
        logger.debug("dismiss_dialog: %s", dialog_state)
        hal.keyboard_controller.key_press("escape")

        # Brief wait for the dialog to close
        time.sleep(_POLL_INTERVAL_S)

        mgr = blackboard.get("_state_manager") or state_manager
        if mgr is not None:
            active = mgr.get_active_states()
            if dialog_state in active:
                raise RuntimeError(
                    f"Dialog state '{dialog_state}' still active after pressing Escape"
                )
            logger.info("Dialog '%s' dismissed successfully", dialog_state)
        else:
            logger.info(
                "Dialog dismiss attempted for '%s' (no state manager to verify)",
                dialog_state,
            )

    return {
        "click_element": click_element,
        "type_text": type_text,
        "navigate_path": navigate_path,
        "navigate_transition": navigate_transition,
        "wait_for_state": wait_for_state,
        "wait_for_element": wait_for_element,
        "dismiss_dialog": dismiss_dialog,
    }
