"""vga.worker._build_runtime wires VgaRuntime to the HAL factory's controllers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from qontinui.vga import worker


def test_build_runtime_uses_hal_factory_controllers():
    mouse, keyboard, capture, client = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    shadow = MagicMock()

    with (
        patch(
            "qontinui.hal.factory.HALFactory.get_mouse_controller", return_value=mouse
        ) as get_mouse,
        patch(
            "qontinui.hal.factory.HALFactory.get_keyboard_controller",
            return_value=keyboard,
        ) as get_keyboard,
        patch(
            "qontinui.hal.implementations.mss_capture.MSSScreenCapture",
            return_value=capture,
        ),
        patch.object(worker, "VgaClient", return_value=client),
        patch.object(worker, "VgaRuntime") as runtime_cls,
    ):
        runtime = worker._build_runtime(shadow_logger=shadow)

    get_mouse.assert_called_once_with()
    get_keyboard.assert_called_once_with()
    runtime_cls.assert_called_once_with(
        client=client,
        hal_mouse=mouse,
        hal_keyboard=keyboard,
        hal_capture=capture,
        shadow_logger=shadow,
    )
    assert runtime is runtime_cls.return_value
