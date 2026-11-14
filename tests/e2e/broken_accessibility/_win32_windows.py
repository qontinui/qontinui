"""Minimal Win32 top-level window lookup via ``ctypes``.

Window lookup for the live E2E harness: enumerate visible
top-level windows (``user32.EnumWindows``), read their titles
(``GetWindowTextW``) and rectangles (``GetWindowRect``), and bring one to
the foreground (``SetForegroundWindow``). Windows-only; every function
returns an empty result / False on any other platform.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WindowInfo:
    """A visible top-level window."""

    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int


def _user32() -> Any:
    """Return ``user32`` with argtypes/restype declared for every call used here."""
    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.EnumWindows.argtypes = [_ENUM_WINDOWS_PROC, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    return user32


if sys.platform == "win32":
    from ctypes import wintypes as _wintypes

    _ENUM_WINDOWS_PROC = ctypes.WINFUNCTYPE(  # type: ignore[attr-defined]
        _wintypes.BOOL, _wintypes.HWND, _wintypes.LPARAM
    )


def find_visible_windows(title_fragment: str) -> list[WindowInfo]:
    """Return visible top-level windows whose title contains ``title_fragment``.

    Matching is case-insensitive. Windows with a zero-width rectangle are
    excluded. An exception raised while visiting a window stops the
    enumeration and is re-raised here rather than swallowed by ctypes.
    """
    if sys.platform != "win32":
        return []

    from ctypes import wintypes

    user32 = _user32()
    needle = title_fragment.lower()
    found: list[WindowInfo] = []
    errors: list[BaseException] = []

    def _visit(hwnd: int, _lparam: int) -> bool:
        try:
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            if needle not in title.lower():
                return True
            rect = wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return True
            width = rect.right - rect.left
            if width > 0:
                found.append(
                    WindowInfo(
                        hwnd=int(hwnd),
                        title=title,
                        left=rect.left,
                        top=rect.top,
                        width=width,
                        height=rect.bottom - rect.top,
                    )
                )
            return True
        except BaseException as exc:  # noqa: BLE001 - re-raised after EnumWindows
            errors.append(exc)
            return False  # stop enumerating

    callback = _ENUM_WINDOWS_PROC(_visit)
    user32.EnumWindows(callback, 0)
    if errors:
        raise errors[0]
    return found


def activate(window: WindowInfo) -> bool:
    """Bring ``window`` to the foreground (best-effort)."""
    if sys.platform != "win32":
        return False
    return bool(_user32().SetForegroundWindow(window.hwnd))
