"""Execution environment detection and configuration.

Automatically detects and configures for different execution environments.
"""

import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution mode enumeration."""

    DEVELOPMENT = "development"
    """Development mode with verbose logging and debugging."""

    TESTING = "testing"
    """Testing mode with test-specific configurations."""

    STAGING = "staging"
    """Staging environment mimicking production."""

    PRODUCTION = "production"
    """Production mode with optimized settings."""

    CI_CD = "ci_cd"
    """Continuous integration/deployment environment."""


class Platform(Enum):
    """Operating system platform."""

    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    UNKNOWN = "unknown"


class DisplayServer(Enum):
    """Display server type."""

    X11 = "x11"
    WAYLAND = "wayland"
    WINDOWS = "windows"
    MACOS = "macos"
    HEADLESS = "headless"
    UNKNOWN = "unknown"


@dataclass
class SystemInfo:
    """System information."""

    platform: Platform
    """Operating system platform."""

    platform_version: str
    """Platform version string."""

    python_version: str
    """Python version."""

    display_server: DisplayServer
    """Display server type."""

    screen_count: int
    """Number of screens/monitors."""

    primary_screen_resolution: tuple
    """Primary screen resolution (width, height)."""

    dpi: int | None
    """Screen DPI if available."""

    is_virtual: bool
    """Whether running in virtual environment."""

    is_container: bool
    """Whether running in container (Docker, etc.)."""

    is_wsl: bool
    """Whether running in Windows Subsystem for Linux."""


class ExecutionEnvironment:
    """Execution environment detection and configuration.

    Automatically detects the execution environment and provides
    appropriate configuration adjustments.

    Features:
    - OS detection (Windows, Linux, macOS)
    - Display server detection (X11, Wayland, etc.)
    - Container/VM detection
    - CI/CD environment detection
    - Screen configuration detection
    - Automatic setting adjustments
    """

    def __init__(self):
        """Initialize and detect environment."""
        self.mode = self._detect_mode()
        self.system_info = self._detect_system_info()
        self._apply_environment_adjustments()

    def _detect_mode(self) -> ExecutionMode:
        """Detect execution mode from environment.

        Returns:
            Detected execution mode
        """
        # Check environment variables
        env_mode = os.environ.get("QONTINUI_MODE", "").lower()

        if env_mode == "production":
            return ExecutionMode.PRODUCTION
        elif env_mode == "staging":
            return ExecutionMode.STAGING
        elif env_mode == "testing":
            return ExecutionMode.TESTING
        elif env_mode == "development":
            return ExecutionMode.DEVELOPMENT

        # Check for CI/CD environments
        ci_vars = [
            "CI",
            "CONTINUOUS_INTEGRATION",
            "GITHUB_ACTIONS",
            "GITLAB_CI",
            "JENKINS",
            "TRAVIS",
            "CIRCLECI",
        ]
        if any(os.environ.get(var) for var in ci_vars):
            return ExecutionMode.CI_CD

        # Check for test runners
        if any(
            mod in os.environ.get("PYTEST_CURRENT_TEST", "")
            for mod in ["pytest", "unittest", "nose"]
        ):
            return ExecutionMode.TESTING

        # Default to development
        return ExecutionMode.DEVELOPMENT

    def _detect_system_info(self) -> SystemInfo:
        """Detect system information.

        Returns:
            System information
        """
        # Detect platform
        system = platform.system().lower()
        if system == "windows":
            platform_enum = Platform.WINDOWS
        elif system == "darwin":
            platform_enum = Platform.MACOS
        elif system == "linux":
            platform_enum = Platform.LINUX
        else:
            platform_enum = Platform.UNKNOWN

        # Detect display server
        display_server = self._detect_display_server(platform_enum)

        # Detect screen info
        screen_count, resolution, dpi = self._detect_screen_info()

        # Detect virtualization
        is_virtual = self._detect_virtualization()
        is_container = self._detect_container()
        is_wsl = self._detect_wsl()

        return SystemInfo(
            platform=platform_enum,
            platform_version=platform.version(),
            python_version=platform.python_version(),
            display_server=display_server,
            screen_count=screen_count,
            primary_screen_resolution=resolution,
            dpi=dpi,
            is_virtual=is_virtual,
            is_container=is_container,
            is_wsl=is_wsl,
        )

    def _detect_display_server(self, platform: Platform) -> DisplayServer:
        """Detect display server type.

        Args:
            platform: Operating system platform

        Returns:
            Display server type
        """
        if platform == Platform.WINDOWS:
            return DisplayServer.WINDOWS
        elif platform == Platform.MACOS:
            return DisplayServer.MACOS
        elif platform == Platform.LINUX:
            # Check for display environment
            if not os.environ.get("DISPLAY"):
                return DisplayServer.HEADLESS

            # Check for Wayland
            if os.environ.get("WAYLAND_DISPLAY"):
                return DisplayServer.WAYLAND

            # Check XDG session type
            session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
            if session_type == "wayland":
                return DisplayServer.WAYLAND
            elif session_type == "x11":
                return DisplayServer.X11

            # Default to X11 if DISPLAY is set
            return DisplayServer.X11

        return DisplayServer.UNKNOWN

    def _detect_screen_info(self) -> tuple:
        """Detect screen configuration.

        Returns:
            Tuple of (screen_count, resolution, dpi)
        """
        try:
            # Try using tkinter (cross-platform)
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()

            screen_count = 1  # tkinter doesn't easily provide multi-monitor info
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            dpi = root.winfo_fpixels("1i")

            root.destroy()

            return screen_count, (width, height), int(dpi)

        except Exception:
            # Fallback to platform-specific methods
            if self.system_info and self.system_info.platform == Platform.LINUX:
                try:
                    # Try xrandr for Linux
                    output = subprocess.check_output(["xrandr"], text=True)
                    lines = output.strip().split("\n")

                    screens = 0
                    primary_res = (1920, 1080)  # Default

                    for line in lines:
                        if " connected" in line:
                            screens += 1
                            if "primary" in line:
                                # Extract resolution
                                parts = line.split()
                                for part in parts:
                                    if "x" in part and part[0].isdigit():
                                        res_parts = part.split("x")
                                        if len(res_parts) == 2:
                                            w = int(res_parts[0])
                                            h = int(res_parts[1].split("+")[0])
                                            primary_res = (w, h)
                                            break

                    return screens, primary_res, None

                except Exception:
                    pass

            # Default fallback
            return 1, (1920, 1080), None

    def _detect_virtualization(self) -> bool:
        """Detect if running in virtual machine.

        Returns:
            True if in VM
        """
        # Check common VM indicators
        vm_indicators = [
            "/proc/sys/hypervisor",  # Linux hypervisor
            "/sys/class/dmi/id/product_name",  # DMI info
        ]

        for indicator in vm_indicators:
            if os.path.exists(indicator):
                try:
                    with open(indicator) as f:
                        content = f.read().lower()
                        if any(
                            vm in content for vm in ["vmware", "virtualbox", "qemu", "kvm", "xen"]
                        ):
                            return True
                except Exception:
                    pass

        # Check environment variables
        if os.environ.get("VIRTUAL_ENV"):
            return True

        return False

    def _detect_container(self) -> bool:
        """Detect if running in container.

        Returns:
            True if in container
        """
        # Check for Docker
        if os.path.exists("/.dockerenv"):
            return True

        # Check for container environment variables
        container_vars = ["DOCKER_CONTAINER", "container", "KUBERNETES_SERVICE_HOST"]
        if any(os.environ.get(var) for var in container_vars):
            return True

        # Check cgroup for container signatures
        try:
            with open("/proc/1/cgroup") as f:
                if "docker" in f.read() or "kubepods" in f.read():
                    return True
        except Exception:
            pass

        return False

    def _detect_wsl(self) -> bool:
        """Detect if running in WSL.

        Returns:
            True if in WSL
        """
        # Check for WSL environment
        if os.environ.get("WSL_DISTRO_NAME"):
            return True

        # Check kernel version
        try:
            with open("/proc/version") as f:
                if "microsoft" in f.read().lower():
                    return True
        except Exception:
            pass

        return False

    def _apply_environment_adjustments(self) -> None:
        """Apply automatic configuration adjustments based on environment."""
        from .framework_settings import get_settings

        settings = get_settings()

        # Headless environment adjustments
        if self.system_info.display_server == DisplayServer.HEADLESS:
            logger.info("Headless environment detected - adjusting settings")
            settings.headless = True
            settings.illustration_enabled = False
            settings.save_snapshots = False

        # Container/CI adjustments
        if self.system_info.is_container or self.mode == ExecutionMode.CI_CD:
            logger.info("Container/CI environment detected - adjusting settings")
            settings.headless = True
            settings.timeout_multiplier = 2.0

        # WSL adjustments
        if self.system_info.is_wsl:
            logger.info("WSL environment detected - adjusting settings")
            # WSL may have display issues
            settings.mouse_move_delay = 1.0

        # Production mode adjustments
        if self.mode == ExecutionMode.PRODUCTION:
            logger.info("Production mode - optimizing settings")
            settings.save_snapshots = False
            settings.collect_dataset = False
            settings.illustration_enabled = False

        # Testing mode adjustments
        elif self.mode == ExecutionMode.TESTING:
            logger.info("Testing mode - enabling test settings")
            settings.mock = True
            settings.timeout_multiplier = 2.0
            settings.screenshot_path = "test_screenshots/"

    def get_info(self) -> dict[str, Any]:
        """Get environment information as dictionary.

        Returns:
            Environment information
        """
        return {
            "mode": self.mode.value,
            "platform": self.system_info.platform.value,
            "platform_version": self.system_info.platform_version,
            "python_version": self.system_info.python_version,
            "display_server": self.system_info.display_server.value,
            "screen_count": self.system_info.screen_count,
            "primary_resolution": self.system_info.primary_screen_resolution,
            "dpi": self.system_info.dpi,
            "is_virtual": self.system_info.is_virtual,
            "is_container": self.system_info.is_container,
            "is_wsl": self.system_info.is_wsl,
        }

    def is_headless(self) -> bool:
        """Check if running in headless environment.

        Returns:
            True if headless
        """
        return self.system_info.display_server == DisplayServer.HEADLESS

    def supports_screenshots(self) -> bool:
        """Check if environment supports screenshots.

        Returns:
            True if screenshots are supported
        """
        return (
            self.system_info.display_server != DisplayServer.HEADLESS
            and not self.system_info.is_container
        )


# Global instance
_environment: ExecutionEnvironment | None = None


def get_environment() -> ExecutionEnvironment:
    """Get the global execution environment.

    Returns:
        ExecutionEnvironment instance
    """
    global _environment
    if _environment is None:
        _environment = ExecutionEnvironment()
    return _environment
