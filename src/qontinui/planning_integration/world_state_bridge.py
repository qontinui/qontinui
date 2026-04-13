"""Bridge between UI Bridge element data and HTN WorldState.

Provides functions to:
1. Query UI Bridge for current element visibility/values
2. Convert to HTN WorldState-compatible dicts
3. Populate blackboard with element data for PlanExecutor
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import TYPE_CHECKING, Any

from multistate.planning.blackboard import Blackboard
from multistate.planning.planner import WorldState
from multistate.planning.world_adapter import WorldStateAdapter

if TYPE_CHECKING:
    from multistate.manager import StateManager

    from qontinui.discovery.target_connection import TargetConnection

logger = logging.getLogger(__name__)

_VALUE_TAG_NAMES: frozenset[str] = frozenset(
    {"input", "textarea", "select", "INPUT", "TEXTAREA", "SELECT"}
)


async def fetch_ui_state(
    connection: TargetConnection,
) -> tuple[dict[str, bool], dict[str, str]]:
    """Query the UI Bridge connection for current element state.

    Args:
        connection: An active UI Bridge target connection.

    Returns:
        A tuple of ``(element_visible, element_values)`` where
        *element_visible* maps element IDs to their visibility boolean and
        *element_values* maps element IDs to their current text/value string.
    """
    elements = await connection.find_elements()

    element_visible: dict[str, bool] = {}
    element_values: dict[str, str] = {}

    for elem in elements:
        element_visible[elem.id] = elem.is_visible

        if elem.tag_name in _VALUE_TAG_NAMES:
            value = elem.attributes.get("value", "")
            if not value and elem.text_content:
                value = elem.text_content
            element_values[elem.id] = value
        elif elem.text_content:
            element_values[elem.id] = elem.text_content

    logger.debug(
        "Fetched UI state: %d elements visible, %d with values",
        len(element_visible),
        len(element_values),
    )
    return element_visible, element_values


def populate_blackboard(
    blackboard: Blackboard,
    element_visible: dict[str, bool],
    element_values: dict[str, str],
    connection: TargetConnection | None = None,
    state_manager: StateManager | None = None,
) -> None:
    """Write UI element data into a :class:`Blackboard`.

    Sets the private keys that :meth:`PlanExecutor._refresh_state` reads:
    ``_ui_elements``, ``_ui_values``, and optionally ``_ui_connection``
    and ``_state_manager``.

    Args:
        blackboard: The blackboard to populate.
        element_visible: Element ID to visibility mapping.
        element_values: Element ID to value mapping.
        connection: Optional UI Bridge connection to store for later refresh.
        state_manager: Optional StateManager to store for later use.
    """
    blackboard.set("_ui_elements", element_visible)
    blackboard.set("_ui_values", element_values)
    if connection is not None:
        blackboard.set("_ui_connection", connection)
    if state_manager is not None:
        blackboard.set("_state_manager", state_manager)


async def create_world_state_snapshot(
    adapter: WorldStateAdapter,
    connection: TargetConnection,
) -> WorldState:
    """Build a :class:`WorldState` populated with live UI Bridge data.

    Args:
        adapter: The world-state adapter backed by a StateManager.
        connection: An active UI Bridge connection.

    Returns:
        A WorldState whose ``element_visible`` and ``element_values``
        reflect the current UI.
    """
    element_visible, element_values = await fetch_ui_state(connection)
    return adapter.snapshot(
        ui_elements=element_visible,
        ui_values=element_values,
    )


def run_async_safe(coro: Any) -> Any:
    """Run an async coroutine from a synchronous context.

    If there is already a running event loop (common inside action handlers
    invoked during plan execution), the coroutine is scheduled via
    :func:`asyncio.ensure_future` and awaited through a
    :class:`concurrent.futures.Future` on a background thread so that the
    caller blocks until the result is ready without deadlocking.

    If no event loop is running, :func:`asyncio.run` is used directly.

    Args:
        coro: An awaitable coroutine.

    Returns:
        The coroutine's return value.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We are inside an already-running loop — run in a thread pool.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


async def refresh_blackboard(blackboard: Blackboard) -> None:
    """Re-fetch UI state and update the blackboard in-place.

    Looks for a ``_ui_connection`` key on the blackboard; if present, calls
    :func:`fetch_ui_state` and writes fresh ``_ui_elements`` /
    ``_ui_values``.

    Args:
        blackboard: The blackboard to refresh.
    """
    connection: TargetConnection | None = blackboard.get("_ui_connection")
    if connection is None:
        logger.debug("refresh_blackboard: no _ui_connection on blackboard, skipping")
        return

    element_visible, element_values = await fetch_ui_state(connection)
    blackboard.set("_ui_elements", element_visible)
    blackboard.set("_ui_values", element_values)
    logger.debug("Refreshed blackboard UI state")
