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


@dataclass(frozen=True)
class WindowInfo:
    """A visible top-level window."""

    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int


def find_visible_windows(title_fragment: str) -> list[WindowInfo]:
    """Return visible top-level windows whose title contains ``title_fragment``.

    Matching is case-insensitive. Windows with a zero-width rectangle are
    excluded.
    """
    if sys.platform != "win32":
        return []

    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    needle = title_fragment.lower()
    found: list[WindowInfo] = []

    enum_proc_type = ctypes.WINFUNCTYPE(  # type: ignore[attr-defined]
        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
    )

    def _visit(hwnd: int, _lparam: int) -> bool:
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
        height = rect.bottom - rect.top
        if width > 0:
            found.append(
                WindowInfo(
                    hwnd=int(hwnd),
                    title=title,
                    left=rect.left,
                    top=rect.top,
                    width=width,
                    height=height,
                )
            )
        return True

    user32.EnumWindows(enum_proc_type(_visit), 0)
    return found


def activate(window: WindowInfo) -> bool:
    """Bring ``window`` to the foreground (best-effort)."""
    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.user32.SetForegroundWindow(window.hwnd))  # type: ignore[attr-defined]
