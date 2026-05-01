"""Event monitor — detects meaningful UI events for background capture triggers.

Monitors for window focus changes and clipboard changes on the host OS.
Used by the BackgroundCaptureService to trigger paired captures only when
something meaningful happens (screenpipe's event-driven pattern).

Platform support:
- Windows: ctypes (GetForegroundWindow, GetWindowText, user32)
- macOS: AppKit (NSWorkspace) via pyobjc (if available)
- Linux: AT-SPI or xdotool (fallback)

Example:
    >>> from qontinui.hal.services.event_monitor import EventMonitor
    >>>
    >>> monitor = EventMonitor()
    >>> info = monitor.get_active_window_info()
    >>> print(f"App: {info.app_name}, Title: {info.window_title}")
    App: Code, Title: paired_capture.py - qontinui
    >>>
    >>> if monitor.has_focus_changed():
    ...     print("Window focus changed!")
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WindowInfo:
    """Information about the currently active window."""

    app_name: str
    """Application/process name (e.g., 'Code', 'Chrome', 'notepad')."""

    window_title: str
    """Window title text."""

    pid: int = 0
    """Process ID of the owning process."""


class EventMonitor:
    """Monitors OS-level UI events for triggering background captures.

    Tracks the currently active window and detects when focus changes.
    This is the "event" part of screenpipe's event-driven capture model.
    """

    def __init__(self) -> None:
        self._last_window: WindowInfo | None = None
        self._last_clipboard: str | None = None
        self._platform = sys.platform

    def get_active_window_info(self) -> WindowInfo:
        """Get information about the currently focused window.

        Returns:
            WindowInfo with app_name and window_title.
            Returns empty WindowInfo if detection fails.
        """
        if self._platform.startswith("win"):
            return self._get_active_window_windows()
        elif self._platform == "darwin":
            return self._get_active_window_macos()
        else:
            return self._get_active_window_linux()

    def has_focus_changed(self) -> bool:
        """Check if the focused window has changed since last check.

        Returns:
            True if the active window is different from the last call.
        """
        current = self.get_active_window_info()
        changed = (
            self._last_window is None
            or current.app_name != self._last_window.app_name
            or current.window_title != self._last_window.window_title
        )
        self._last_window = current
        return changed

    def has_clipboard_changed(self) -> bool:
        """Check if clipboard content has changed since last check.

        Returns:
            True if clipboard text differs from the last check.
            Returns False if clipboard cannot be read.
        """
        try:
            current = self._get_clipboard_text()
            changed = current != self._last_clipboard
            self._last_clipboard = current
            return changed
        except Exception:
            return False

    # ========================================================================
    # Platform-specific implementations
    # ========================================================================

    def _get_active_window_windows(self) -> WindowInfo:
        """Windows implementation using ctypes."""
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]

            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return WindowInfo(app_name="", window_title="")

            # Get window title
            length = user32.GetWindowTextLengthW(hwnd) + 1
            buf = ctypes.create_unicode_buffer(length)
            user32.GetWindowTextW(hwnd, buf, length)
            window_title = buf.value

            # Get process ID
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

            # Get process name from PID
            app_name = self._get_process_name_windows(pid.value)

            return WindowInfo(
                app_name=app_name, window_title=window_title, pid=pid.value
            )
        except Exception as e:
            logger.debug("Windows active window detection failed: %s", e)
            return WindowInfo(app_name="", window_title="")

    @staticmethod
    def _get_process_name_windows(pid: int) -> str:
        """Get process name from PID on Windows."""
        try:
            import ctypes

            PROCESS_QUERY_INFORMATION = 0x0400
            PROCESS_VM_READ = 0x0010

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            psapi = ctypes.windll.psapi  # type: ignore[attr-defined]

            handle = kernel32.OpenProcess(
                PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid
            )
            if not handle:
                return ""

            try:
                buf = ctypes.create_unicode_buffer(260)
                psapi.GetModuleBaseNameW(handle, None, buf, 260)
                name = buf.value
                # Strip .exe suffix
                if name.lower().endswith(".exe"):
                    name = name[:-4]
                return name
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return ""

    def _get_active_window_macos(self) -> WindowInfo:
        """macOS implementation using AppKit."""
        try:
            from AppKit import NSWorkspace  # type: ignore[import-not-found]

            workspace = NSWorkspace.sharedWorkspace()
            active_app = workspace.activeApplication()
            if active_app:
                return WindowInfo(
                    app_name=active_app.get("NSApplicationName", ""),
                    window_title=active_app.get("NSApplicationName", ""),
                    pid=active_app.get("NSApplicationProcessIdentifier", 0),
                )
        except ImportError:
            logger.debug("AppKit not available for macOS window detection")
        except Exception as e:
            logger.debug("macOS active window detection failed: %s", e)
        return WindowInfo(app_name="", window_title="")

    def _get_active_window_linux(self) -> WindowInfo:
        """Linux implementation using xdotool."""
        try:
            import subprocess

            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                title = result.stdout.strip()
                # Try to get process name
                wid_result = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowpid"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                pid = (
                    int(wid_result.stdout.strip()) if wid_result.returncode == 0 else 0
                )
                app_name = self._get_process_name_linux(pid) if pid else ""
                return WindowInfo(app_name=app_name, window_title=title, pid=pid)
        except FileNotFoundError:
            logger.debug("xdotool not found for Linux window detection")
        except Exception as e:
            logger.debug("Linux active window detection failed: %s", e)
        return WindowInfo(app_name="", window_title="")

    @staticmethod
    def _get_process_name_linux(pid: int) -> str:
        """Get process name from PID on Linux."""
        try:
            with open(f"/proc/{pid}/comm") as f:
                return f.read().strip()
        except Exception:
            return ""

    def _get_clipboard_text(self) -> str | None:
        """Get current clipboard text content (cross-platform)."""
        if self._platform.startswith("win"):
            return self._get_clipboard_windows()
        elif self._platform == "darwin":
            return self._get_clipboard_macos()
        else:
            return self._get_clipboard_linux()

    @staticmethod
    def _get_clipboard_windows() -> str | None:
        """Windows clipboard via ctypes."""
        try:
            import ctypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

            CF_UNICODETEXT = 13

            if not user32.OpenClipboard(0):
                return None
            try:
                handle = user32.GetClipboardData(CF_UNICODETEXT)
                if not handle:
                    return None
                ptr = kernel32.GlobalLock(handle)
                if not ptr:
                    return None
                try:
                    return ctypes.wstring_at(ptr)  # type: ignore[attr-defined]
                finally:
                    kernel32.GlobalUnlock(handle)
            finally:
                user32.CloseClipboard()
        except Exception:
            return None

    @staticmethod
    def _get_clipboard_macos() -> str | None:
        """macOS clipboard via pbpaste."""
        try:
            import subprocess

            result = subprocess.run(
                ["pbpaste"], capture_output=True, text=True, timeout=2
            )
            return result.stdout if result.returncode == 0 else None
        except Exception:
            return None

    @staticmethod
    def _get_clipboard_linux() -> str | None:
        """Linux clipboard via xclip."""
        try:
            import subprocess

            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            return result.stdout if result.returncode == 0 else None
        except Exception:
            return None
