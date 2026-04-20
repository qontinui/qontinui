"""Tests for world_state_bridge module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from multistate.planning.blackboard import Blackboard
from multistate.planning.planner import WorldState
from multistate.planning.world_adapter import WorldStateAdapter

from qontinui.discovery.target_connection import Element
from qontinui.planning_integration.world_state_bridge import (
    create_world_state_snapshot,
    fetch_ui_state,
    populate_blackboard,
    refresh_blackboard,
)


def _make_element(
    id: str,
    tag_name: str,
    is_visible: bool = True,
    is_enabled: bool = True,
    text_content: str | None = None,
    attributes: dict[str, str] | None = None,
) -> Element:
    """Build a minimal Element for testing."""
    return Element(
        id=id,
        tag_name=tag_name,
        is_visible=is_visible,
        is_enabled=is_enabled,
        text_content=text_content,
        attributes=attributes or {},
    )


def _mock_connection_with_three_elements() -> AsyncMock:
    """Return a mock connection with three elements."""
    conn = AsyncMock()
    conn.find_elements.return_value = [
        _make_element(
            id="btn-submit",
            tag_name="button",
            is_visible=True,
            text_content="Submit",
        ),
        _make_element(
            id="input-email",
            tag_name="input",
            is_visible=False,
            attributes={"value": "user@example.com"},
        ),
        _make_element(
            id="txt-heading",
            tag_name="h1",
            is_visible=True,
            is_enabled=True,
            text_content="Welcome",
        ),
    ]
    return conn


# ------------------------------------------------------------------
# test_fetch_ui_state
# ------------------------------------------------------------------


def test_fetch_ui_state() -> None:
    """Visibility and value dicts are built correctly from elements."""
    conn = _mock_connection_with_three_elements()
    visible, values = asyncio.run(fetch_ui_state(conn))

    # All three elements should be in visible dict
    assert visible == {
        "btn-submit": True,
        "input-email": False,
        "txt-heading": True,
    }

    # input has value attribute, button has text_content, heading has text_content
    assert values["input-email"] == "user@example.com"
    assert values["btn-submit"] == "Submit"
    assert values["txt-heading"] == "Welcome"


def test_fetch_ui_state_input_fallback_to_text_content() -> None:
    """Input element without value attribute falls back to text_content."""
    conn = AsyncMock()
    conn.find_elements.return_value = [
        _make_element(
            id="textarea-1",
            tag_name="textarea",
            is_visible=True,
            text_content="Some typed text",
            attributes={},
        ),
    ]
    visible, values = asyncio.run(fetch_ui_state(conn))
    assert visible == {"textarea-1": True}
    assert values["textarea-1"] == "Some typed text"


# ------------------------------------------------------------------
# test_populate_blackboard
# ------------------------------------------------------------------


def test_populate_blackboard() -> None:
    """All keys are written correctly to the blackboard."""
    bb = Blackboard()
    vis = {"btn-1": True}
    vals = {"btn-1": "OK"}
    fake_conn = MagicMock()
    fake_mgr = MagicMock()

    populate_blackboard(bb, vis, vals, connection=fake_conn, state_manager=fake_mgr)

    assert bb.get("_ui_elements") == vis
    assert bb.get("_ui_values") == vals
    assert bb.get("_ui_connection") is fake_conn
    assert bb.get("_state_manager") is fake_mgr


def test_populate_blackboard_optional_args() -> None:
    """Connection and state_manager are optional."""
    bb = Blackboard()
    populate_blackboard(bb, {"x": False}, {"x": ""})

    assert bb.get("_ui_elements") == {"x": False}
    assert bb.get("_ui_values") == {"x": ""}
    assert bb.get("_ui_connection") is None
    assert bb.get("_state_manager") is None


# ------------------------------------------------------------------
# test_create_world_state_snapshot
# ------------------------------------------------------------------


def test_create_world_state_snapshot() -> None:
    """Snapshot integrates UI element data into the WorldState."""
    conn = _mock_connection_with_three_elements()

    manager = MagicMock()
    manager.get_active_states.return_value = {"state_home"}
    manager.get_available_transitions.return_value = ["t1"]

    adapter = WorldStateAdapter(manager)

    ws = asyncio.run(create_world_state_snapshot(adapter, conn))

    assert isinstance(ws, WorldState)
    assert ws.active_states == {"state_home"}
    assert "t1" in ws.available_transitions
    # Element data should be present
    assert ws.element_visible["btn-submit"] is True
    assert ws.element_visible["input-email"] is False
    assert ws.element_values["input-email"] == "user@example.com"


# ------------------------------------------------------------------
# test_refresh_blackboard
# ------------------------------------------------------------------


def test_refresh_blackboard() -> None:
    """Refresh fetches fresh data and updates blackboard keys."""
    conn = AsyncMock()
    # Initial elements
    conn.find_elements.return_value = [
        _make_element(id="a", tag_name="button", is_visible=True, text_content="A"),
    ]

    bb = Blackboard()
    populate_blackboard(bb, {"old": True}, {"old": "stale"}, connection=conn)

    # Now the connection returns different elements
    conn.find_elements.return_value = [
        _make_element(id="b", tag_name="input", is_visible=False, attributes={"value": "fresh"}),
    ]

    asyncio.run(refresh_blackboard(bb))

    updated_vis = bb.get("_ui_elements")
    updated_vals = bb.get("_ui_values")
    assert updated_vis == {"b": False}
    assert updated_vals == {"b": "fresh"}


def test_refresh_blackboard_no_connection() -> None:
    """Refresh is a no-op when no connection is stored."""
    bb = Blackboard()
    bb.set("_ui_elements", {"x": True})

    asyncio.run(refresh_blackboard(bb))

    # Should be unchanged
    assert bb.get("_ui_elements") == {"x": True}
