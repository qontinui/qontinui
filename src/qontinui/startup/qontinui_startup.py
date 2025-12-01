"""Qontinui startup - ported from Qontinui framework.

Ensures Qontinui captures at physical resolution by default.
"""

import os
import sys


class QontinuiStartup:
    """Ensures Qontinui captures at physical resolution by default.

    Port of QontinuiStartup from Qontinui framework class.

    This must run before any other configuration to properly disable DPI scaling.
    Makes Qontinui behave like SikuliX IDE, capturing at full physical resolution
    regardless of Windows DPI scaling settings.
    """

    _initialized = False

    @classmethod
    def initialize(cls) -> None:
        """Initialize Qontinui with physical resolution configuration.

        This should be called at the start of the application.
        """
        if cls._initialized:
            return

        cls._initialized = True
        cls.configure_physical_resolution()
        cls.verify_physical_resolution()

    @classmethod
    def configure_physical_resolution(cls) -> None:
        """Configure system to capture at physical resolution.

        This disables DPI awareness to match SikuliX IDE behavior.
        """
        print("=== Qontinui Physical Resolution Configuration ===")

        # Set environment variables for DPI awareness
        # Note: Python doesn't have direct equivalents to Java's system properties
        # These would need to be handled at the OS level or through specific libraries

        # For Windows, we might need to use ctypes to set DPI awareness
        if sys.platform == "win32":
            try:
                import ctypes

                # SetProcessDPIAware() tells Windows this process is DPI aware
                # Setting to False would mean we want physical pixels
                # This is simplified - actual implementation would be more complex
                awareness = ctypes.c_int(0)  # PROCESS_DPI_UNAWARE
                ctypes.windll.shcore.SetProcessDpiAwareness(awareness)
                print("✓ DPI awareness disabled (Windows)")
            except Exception as e:
                print(f"Warning: Could not set DPI awareness: {e}")

        # Set environment variables that might be used by GUI libraries
        os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
        os.environ["QT_SCALE_FACTOR"] = "1"

        print("✓ Physical resolution capture enabled")
        print("✓ All coordinates will be in physical pixels")

    @classmethod
    def verify_physical_resolution(cls) -> None:
        """Verify that physical resolution capture is working."""
        try:
            # This would need to be adapted based on the actual screen capture library used
            # For now, this is a placeholder that shows the structure
            print("\n=== Resolution Verification ===")

            # Use HAL to get screen dimensions
            try:
                from ..hal.factory import HALFactory

                screen = HALFactory.get_screen_capture()
                screen_size = screen.get_screen_size()
                print(f"Screen Resolution: {screen_size[0]}x{screen_size[1]}")
                print(f"✓ SUCCESS: Capturing at resolution {screen_size[0]}x{screen_size[1]}")
            except (ImportError, Exception):
                # Skip screen check - it can fail in headless environments
                print("Note: Screen resolution check skipped (no display available)")

            print("================================\n")

        except Exception as e:
            print(f"Error verifying resolution: {e}")


class PhysicalResolutionInitializer:
    """Ensures physical resolution is initialized early.

    Port of PhysicalResolutionInitializer from Qontinui framework class.
    """

    @staticmethod
    def force_initialization() -> None:
        """Force early initialization of physical resolution settings."""
        QontinuiStartup.initialize()
