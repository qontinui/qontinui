"""Unit tests for the pynput-backed HAL input operations (mocked controllers)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pynput.keyboard import Key as PynputKey

from qontinui.hal.implementations.keyboard_operations import (
    PYAUTOGUI_KEY_ALIASES,
    KeyboardOperations,
)
from qontinui.hal.implementations.mouse_operations import MouseOperations


class TestMouseScroll:
    """MouseOperations.mouse_scroll drives pynput's scroll(dx, dy)."""

    def test_vertical_scroll_has_no_horizontal_component(self):
        controller = MagicMock()
        ops = MouseOperations(controller)

        assert ops.mouse_scroll(5) is True

        controller.scroll.assert_called_once_with(0, 5)

    @pytest.mark.parametrize(
        ("clicks", "horizontal", "expected"),
        [(0, 4, (4, 0)), (0, -3, (-3, 0)), (-2, 7, (7, -2))],
    )
    def test_horizontal_clicks_are_dx(self, clicks, horizontal, expected):
        controller = MagicMock()
        ops = MouseOperations(controller)

        assert ops.mouse_scroll(clicks, horizontal_clicks=horizontal) is True

        controller.scroll.assert_called_once_with(*expected)


class TestPyautoguiKeyAliases:
    """pyautogui key names resolve to the pynput keys they meant."""

    @pytest.mark.parametrize(("name", "attr"), sorted(PYAUTOGUI_KEY_ALIASES.items()))
    def test_alias_resolves_to_pynput_key(self, name, attr):
        expected = getattr(PynputKey, attr, None)
        if expected is None:
            pytest.skip(f"pynput has no Key.{attr} on this platform")
        ops = KeyboardOperations(MagicMock())

        assert ops._get_pynput_key(name) == expected
        assert ops._get_pynput_key(name.upper()) == expected

    def test_hotkey_presses_aliased_keys(self):
        controller = MagicMock()
        ops = KeyboardOperations(controller)

        assert ops.hotkey("command", "esc") is True

        pressed = [c.args[0] for c in controller.press.call_args_list]
        assert pressed == [PynputKey.cmd, PynputKey.esc]

    def test_existing_enum_names_still_win(self):
        ops = KeyboardOperations(MagicMock())

        assert ops._get_pynput_key("escape") == PynputKey.esc
        assert ops._get_pynput_key("pagedown") == PynputKey.page_down

    def test_plain_characters_pass_through(self):
        ops = KeyboardOperations(MagicMock())

        assert ops._get_pynput_key("a") == "a"
