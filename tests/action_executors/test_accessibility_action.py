"""Tests for accessibility-aware action dispatch helper."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from qontinui.action_executors.accessibility_action import (
    try_accessibility_click,
    try_accessibility_type,
)


def _make_action_result(ref=None, role=None):
    """Build a mock last_action_result with accessibility backend_metadata."""
    backend_metadata = {}
    if ref is not None:
        backend_metadata["ref"] = ref
    if role is not None:
        backend_metadata["role"] = role

    metadata = MagicMock()
    metadata.backend_metadata = backend_metadata

    match_obj = MagicMock()
    match_obj.metadata = metadata

    match = MagicMock()
    match.match_object = match_obj
    match.region = None

    result = MagicMock()
    result.matches = (match,)
    return result


def _make_hal_container(connected=True, has_dispatch=False):
    """Build a mock HAL container with accessibility capture."""
    capture = MagicMock()
    capture.is_connected.return_value = connected
    capture.click_by_ref = AsyncMock(return_value=True)
    capture.type_by_ref = AsyncMock(return_value=True)
    capture.perform_action = AsyncMock(return_value=True)

    container = MagicMock()
    container.accessibility_capture = capture
    container.action_dispatch = None

    if has_dispatch:
        try:
            from qontinui.hal.implementations.accessibility.action_dispatch import (
                ActionDispatchRegistry,
            )

            container.action_dispatch = ActionDispatchRegistry()
        except ImportError:
            pass

    return container


# ---------------------------------------------------------------------------
# try_accessibility_click guard conditions
# ---------------------------------------------------------------------------


class TestClickGuards:
    """Verify all guard conditions return handled=False."""

    def test_none_action_result(self):
        r = asyncio.run(try_accessibility_click(None, MagicMock()))
        assert not r.handled

    def test_empty_matches(self):
        ar = MagicMock()
        ar.matches = ()
        r = asyncio.run(try_accessibility_click(ar, MagicMock()))
        assert not r.handled

    def test_no_ref_in_metadata(self):
        ar = _make_action_result(ref=None)
        r = asyncio.run(try_accessibility_click(ar, MagicMock()))
        assert not r.handled

    def test_none_hal_container(self):
        ar = _make_action_result(ref="@e1", role="button")
        r = asyncio.run(try_accessibility_click(ar, None))
        assert not r.handled

    def test_capture_not_connected(self):
        ar = _make_action_result(ref="@e1", role="button")
        hal = _make_hal_container(connected=False)
        r = asyncio.run(try_accessibility_click(ar, hal))
        assert not r.handled

    def test_no_capture_on_container(self):
        ar = _make_action_result(ref="@e1", role="button")
        hal = MagicMock()
        hal.accessibility_capture = None
        r = asyncio.run(try_accessibility_click(ar, hal))
        assert not r.handled


# ---------------------------------------------------------------------------
# try_accessibility_click happy path
# ---------------------------------------------------------------------------


class TestClickSuccess:
    def test_click_by_ref_fallback(self):
        """When no dispatch registry, falls back to click_by_ref."""
        ar = _make_action_result(ref="@e1", role="button")
        hal = _make_hal_container(connected=True, has_dispatch=False)
        r = asyncio.run(try_accessibility_click(ar, hal))
        assert r.handled
        assert r.success
        hal.accessibility_capture.click_by_ref.assert_called_once_with("@e1")

    def test_dispatch_via_registry(self):
        """When dispatch registry available, uses pattern strategy."""
        ar = _make_action_result(ref="@e1", role="button")
        hal = _make_hal_container(connected=True, has_dispatch=True)
        r = asyncio.run(try_accessibility_click(ar, hal))
        assert r.handled
        assert r.success

    def test_click_by_ref_failure(self):
        """click_by_ref returning False → success=False but handled=True."""
        ar = _make_action_result(ref="@e1", role="button")
        hal = _make_hal_container(connected=True, has_dispatch=False)
        hal.accessibility_capture.click_by_ref = AsyncMock(return_value=False)
        r = asyncio.run(try_accessibility_click(ar, hal))
        assert r.handled
        assert not r.success


# ---------------------------------------------------------------------------
# try_accessibility_type guard conditions
# ---------------------------------------------------------------------------


class TestTypeGuards:
    def test_none_action_result(self):
        r = asyncio.run(try_accessibility_type(None, MagicMock(), "hello"))
        assert not r.handled

    def test_no_ref(self):
        ar = _make_action_result(ref=None)
        r = asyncio.run(try_accessibility_type(ar, MagicMock(), "hello"))
        assert not r.handled


# ---------------------------------------------------------------------------
# try_accessibility_type happy path
# ---------------------------------------------------------------------------


class TestTypeSuccess:
    def test_type_by_ref_fallback(self):
        """When no dispatch registry, falls back to type_by_ref."""
        ar = _make_action_result(ref="@e1", role="textbox")
        hal = _make_hal_container(connected=True, has_dispatch=False)
        r = asyncio.run(try_accessibility_type(ar, hal, "hello"))
        assert r.handled
        assert r.success

    def test_dispatch_via_registry(self):
        """When dispatch registry available, uses TextBoxStrategy."""
        ar = _make_action_result(ref="@e1", role="textbox")
        hal = _make_hal_container(connected=True, has_dispatch=True)
        r = asyncio.run(try_accessibility_type(ar, hal, "hello"))
        assert r.handled
        assert r.success

    def test_type_by_ref_failure(self):
        ar = _make_action_result(ref="@e1", role="textbox")
        hal = _make_hal_container(connected=True, has_dispatch=False)
        hal.accessibility_capture.type_by_ref = AsyncMock(return_value=False)
        r = asyncio.run(try_accessibility_type(ar, hal, "hello"))
        assert r.handled
        assert not r.success
