"""Linux platform-specific implementation (stub)."""

import sys

from ...config import HALConfig
from ...interfaces.platform_specific import IPlatformSpecific, UIElement, Window


class LinuxPlatform(IPlatformSpecific):
    """Stub Linux platform implementation.

    This is a minimal stub to allow testing. Full implementation pending.
    """

    def __init__(self, config: HALConfig) -> None:
        """Initialize Linux platform."""
        self.config = config

    def get_all_windows(self) -> list[Window]:
        """Get list of all windows."""
        return []

    def get_window_by_title(self, title: str, partial: bool = False) -> Window | None:
        """Find window by title."""
        return None

    def get_window_by_process(self, process_name: str) -> list[Window]:
        """Get windows belonging to a process."""
        return []

    def get_active_window(self) -> Window | None:
        """Get currently active/focused window."""
        return None

    def set_window_focus(self, window: Window) -> bool:
        """Set focus to window."""
        return False

    def minimize_window(self, window: Window) -> bool:
        """Minimize window."""
        return False

    def maximize_window(self, window: Window) -> bool:
        """Maximize window."""
        return False

    def restore_window(self, window: Window) -> bool:
        """Restore window to normal size."""
        return False

    def move_window(self, window: Window, x: int, y: int) -> bool:
        """Move window to position."""
        return False

    def resize_window(self, window: Window, width: int, height: int) -> bool:
        """Resize window."""
        return False

    def get_ui_elements(self, window: Window) -> list[UIElement]:
        """Get all UI elements in window."""
        return []

    def find_ui_element(
        self,
        window: Window,
        name: str | None = None,
        type: str | None = None,
        text: str | None = None,
    ) -> UIElement | None:
        """Find UI element by properties."""
        return None

    def click_ui_element(self, element: UIElement) -> bool:
        """Click UI element using platform API."""
        return False

    def set_ui_text(self, element: UIElement, text: str) -> bool:
        """Set text in UI element."""
        return False

    def get_ui_text(self, element: UIElement) -> str | None:
        """Get text from UI element."""
        return None

    def get_platform_name(self) -> str:
        """Get platform name."""
        return "Linux"

    def get_platform_version(self) -> str:
        """Get platform version."""
        return sys.version

    def get_screen_resolution(self) -> tuple[int, int]:
        """Get primary screen resolution."""
        return (1920, 1080)

    def get_dpi_scaling(self) -> float:
        """Get DPI scaling factor."""
        return 1.0

    def is_dark_mode(self) -> bool:
        """Check if system is in dark mode."""
        return False
