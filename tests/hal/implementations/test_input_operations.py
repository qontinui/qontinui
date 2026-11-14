"""Unit tests for the pynput-backed HAL input operations (mocked controllers)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pynput.keyboard import Key as PynputKey

from qontinui.exceptions import InputControlError
from qontinui.hal.implementations import keyboard_operations
from qontinui.hal.implementations.keyboard_operations import (
    PYAUTOGUI_KEY_ALIASES,
    KeyboardOperations,
)
from qontinui.hal.implementations.mouse_operations import MouseOperations


class _PynputLikeController:
    """Stub keyboard controller reproducing pynput's key resolution rule.

    pynput's ``Controller._resolve`` raises ``ValueError(key)`` for a string
    whose length is not 1; Key members and single characters are accepted.
    Records presses and releases; needs no display.
    """

    def __init__(self, fail_on_press: object = None, error: Exception | None = None):
        self.events: list[tuple[str, object]] = []
        self._fail_on_press = fail_on_press
        self._error = error

    def _resolve(self, key: object) -> None:
        if isinstance(key, str) and len(key) != 1:
            raise ValueError(key)

    def press(self, key: object) -> None:
        self._resolve(key)
        if self._error is not None and key == self._fail_on_press:
            raise self._error
        self.events.append(("press", key))

    def release(self, key: object) -> None:
        self._resolve(key)
        self.events.append(("release", key))


class _InvalidKeyLikeError(Exception):
    """Stands in for pynput's InvalidKeyException (an Exception subclass)."""


class TestHotkeyReleases:
    """hotkey leaves no key held down, whatever fails."""

    def test_success_presses_then_releases_in_reverse(self):
        controller = _PynputLikeController()

        assert KeyboardOperations(controller).hotkey("ctrl", "shift", "a") is True

        assert controller.events == [
            ("press", PynputKey.ctrl),
            ("press", PynputKey.shift),
            ("press", "a"),
            ("release", "a"),
            ("release", PynputKey.shift),
            ("release", PynputKey.ctrl),
        ]

    def test_non_valueerror_mid_hotkey_releases_pressed_keys(self):
        controller = _PynputLikeController(
            fail_on_press="a", error=_InvalidKeyLikeError("invalid key")
        )

        with pytest.raises(InputControlError):
            KeyboardOperations(controller).hotkey("ctrl", "shift", "a", "b")

        assert controller.events == [
            ("press", PynputKey.ctrl),
            ("press", PynputKey.shift),
            ("release", PynputKey.shift),
            ("release", PynputKey.ctrl),
        ]

    def test_unresolvable_key_releases_modifiers(self):
        controller = _PynputLikeController()

        with pytest.raises(InputControlError):
            KeyboardOperations(controller).hotkey("ctrl", "num0")

        assert controller.events == [
            ("press", PynputKey.ctrl),
            ("release", PynputKey.ctrl),
        ]

    def test_release_failure_still_releases_the_rest(self):
        controller = _PynputLikeController()
        original_release = controller.release

        def release(key: object) -> None:
            if key == PynputKey.shift:
                raise _InvalidKeyLikeError("stuck")
            original_release(key)

        controller.release = release  # type: ignore[method-assign]

        with pytest.raises(InputControlError):
            KeyboardOperations(controller).hotkey("ctrl", "shift", "a")

        assert ("release", "a") in controller.events
        assert ("release", PynputKey.ctrl) in controller.events


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

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("return", PynputKey.enter),
            ("esc", PynputKey.esc),
            ("pgdn", PynputKey.page_down),
            ("pgup", PynputKey.page_up),
            ("del", PynputKey.delete),
            ("command", PynputKey.cmd),
            ("option", PynputKey.alt),
            ("control", PynputKey.ctrl),
            ("winleft", PynputKey.cmd_l),
            ("capslock", PynputKey.caps_lock),
            ("prtsc", PynputKey.print_screen),
            ("f13", PynputKey.f13),
        ],
    )
    def test_literal_alias_pairs(self, name, expected):
        ops = KeyboardOperations(MagicMock())

        assert ops._get_pynput_key(name) == expected
        assert ops._get_pynput_key(name.upper()) == expected

    @pytest.mark.parametrize(("name", "attr"), sorted(PYAUTOGUI_KEY_ALIASES.items()))
    def test_every_alias_names_a_pynput_key_here(self, name, attr):
        """On this (Linux) platform every alias target exists in pynput."""
        ops = KeyboardOperations(MagicMock())

        assert ops._get_pynput_key(name) == getattr(PynputKey, attr)

    def test_alias_falls_through_when_platform_lacks_the_key(self, monkeypatch):
        """A key the platform's pynput lacks resolves to the raw name."""
        ops = KeyboardOperations(MagicMock())
        monkeypatch.setattr(keyboard_operations, "PynputKey", SimpleNamespace(esc=PynputKey.esc))

        assert ops._get_pynput_key("insert") == "insert"
        assert ops._get_pynput_key("esc") == PynputKey.esc

    def test_print_is_not_print_screen(self):
        """'print' is VK_PRINT on Windows, not Print Screen: no alias."""
        assert "print" not in PYAUTOGUI_KEY_ALIASES
        assert KeyboardOperations(MagicMock())._get_pynput_key("print") == "print"

    @pytest.mark.parametrize(
        "name",
        ["num0", "num9", "add", "subtract", "multiply", "divide", "decimal", "separator"],
    )
    def test_keypad_names_fail_loudly(self, name):
        """Keypad names have no pynput key: a hotkey using one raises."""
        ops = KeyboardOperations(_PynputLikeController())

        assert ops._get_pynput_key(name) == name
        with pytest.raises(InputControlError):
            ops.hotkey(name)

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
