"""HAL initialization and lifecycle management.

This module provides functions for initializing and shutting down HAL
components using explicit dependency injection instead of global factories.
"""

import sys
from typing import cast

from .config import HALConfig
from .container import HALContainer
from .interfaces import (
    IInputController,
    IOCREngine,
    IPatternMatcher,
    IPlatformSpecific,
    IScreenCapture,
)


class HALInitializationError(Exception):
    """Raised when HAL component initialization fails."""

    pass


def initialize_hal(config: HALConfig | None = None) -> HALContainer:
    """Initialize HAL components and return container.

    This function creates all HAL component instances based on configuration
    and returns them in a container. This should be called once at application
    startup.

    All initialization happens eagerly and synchronously. Any errors are
    raised immediately, implementing a fail-fast approach.

    Args:
        config: HAL configuration. If None, uses default config.

    Returns:
        HALContainer with all components initialized

    Raises:
        HALInitializationError: If any component fails to initialize
        ImportError: If required backend library is missing
        ValueError: If configuration is invalid

    Example:
        >>> from qontinui.hal import initialize_hal, shutdown_hal
        >>> # At application startup
        >>> hal = initialize_hal()
        >>> try:
        ...     # Use hal throughout application
        ...     executor = ActionExecutor(config, hal=hal)
        ...     executor.execute_action(action)
        ... finally:
        ...     shutdown_hal(hal)
    """
    if config is None:
        config = HALConfig()

    # Validate configuration first
    try:
        config.validate()
    except ValueError as e:
        raise HALInitializationError(f"Invalid HAL configuration: {e}") from e

    # Create container with all components
    try:
        container = HALContainer.create_from_config(config)
    except ImportError as e:
        raise HALInitializationError(
            f"Failed to import HAL backend: {e}. "
            f"Make sure required libraries are installed."
        ) from e
    except Exception as e:
        raise HALInitializationError(f"Failed to initialize HAL components: {e}") from e

    return container


def shutdown_hal(container: HALContainer) -> None:
    """Shutdown HAL components and release resources.

    This function should be called when the application is shutting down
    to properly clean up HAL component resources.

    Args:
        container: HAL container to shutdown

    Example:
        >>> hal = initialize_hal()
        >>> try:
        ...     # Use hal
        ...     pass
        ... finally:
        ...     shutdown_hal(hal)
    """
    if container:
        container.cleanup()


# Internal component creation functions
# These are used by HALContainer.create_from_config()


def _detect_platform(config: HALConfig) -> str:
    """Detect current platform.

    Args:
        config: HAL configuration

    Returns:
        Platform name ('windows', 'macos', 'linux')

    Raises:
        ValueError: If platform cannot be detected
    """
    if config.platform_override != "auto":
        return config.platform_override

    if sys.platform.startswith("win"):
        return "windows"
    elif sys.platform == "darwin":
        return "macos"
    elif sys.platform.startswith("linux"):
        return "linux"
    else:
        raise ValueError(f"Unsupported platform: {sys.platform}")


def _create_screen_capture(config: HALConfig) -> IScreenCapture:
    """Create screen capture implementation.

    Args:
        config: HAL configuration

    Returns:
        Screen capture implementation

    Raises:
        ImportError: If backend is not available
        ValueError: If backend is not supported
    """
    backend = config.capture_backend.lower()

    if backend == "mss":
        from .implementations.mss_capture import MSSScreenCapture

        return MSSScreenCapture(config)

    elif backend == "pyautogui":
        from .implementations.pyautogui_capture import PyAutoGUIScreenCapture

        return PyAutoGUIScreenCapture(config)

    elif backend == "pillow":
        from .implementations.pillow_capture import PillowScreenCapture

        return PillowScreenCapture(config)

    elif backend == "native":
        return _create_native_screen_capture(config)

    else:
        raise ValueError(f"Unsupported screen capture backend: {backend}")


def _create_native_screen_capture(config: HALConfig) -> IScreenCapture:
    """Create native screen capture for current platform.

    Args:
        config: HAL configuration

    Returns:
        Native screen capture implementation

    Raises:
        ValueError: If no native implementation for platform
    """
    platform = _detect_platform(config)

    if platform == "windows":
        from .implementations.platform.windows_capture import WindowsScreenCapture

        return cast(IScreenCapture, WindowsScreenCapture(config))

    elif platform == "macos":
        from .implementations.platform.macos_capture import MacOSScreenCapture

        return cast(IScreenCapture, MacOSScreenCapture(config))

    elif platform == "linux":
        from .implementations.platform.linux_capture import LinuxScreenCapture

        return cast(IScreenCapture, LinuxScreenCapture(config))

    else:
        raise ValueError(f"No native screen capture for platform: {platform}")


def _create_pattern_matcher(config: HALConfig) -> IPatternMatcher:
    """Create pattern matcher implementation.

    Args:
        config: HAL configuration

    Returns:
        Pattern matcher implementation

    Raises:
        ImportError: If backend is not available
        ValueError: If backend is not supported
    """
    backend = config.matcher_backend.lower()

    if backend == "opencv":
        from .implementations.opencv_matcher import OpenCVMatcher

        return OpenCVMatcher(config)

    elif backend == "pyautogui":
        from .implementations.pyautogui_matcher import PyAutoGUIMatcher

        return PyAutoGUIMatcher(config)

    elif backend == "tensorflow":
        from .implementations.tensorflow_matcher import TensorFlowMatcher

        return TensorFlowMatcher(config)

    elif backend == "native":
        return _create_native_pattern_matcher(config)

    else:
        raise ValueError(f"Unsupported pattern matcher backend: {backend}")


def _create_native_pattern_matcher(config: HALConfig) -> IPatternMatcher:
    """Create native pattern matcher for current platform.

    Args:
        config: HAL configuration

    Returns:
        Native pattern matcher implementation
    """
    # Most platforms use OpenCV as "native"
    from .implementations.opencv_matcher import OpenCVMatcher

    return OpenCVMatcher(config)


def _create_input_controller(config: HALConfig) -> IInputController:
    """Create input controller implementation.

    Args:
        config: HAL configuration

    Returns:
        Input controller implementation

    Raises:
        ImportError: If backend is not available
        ValueError: If backend is not supported
    """
    backend = config.input_backend.lower()

    if backend == "pynput":
        from .implementations.pynput_controller import PynputController

        return PynputController(config)

    elif backend == "pyautogui":
        from .implementations.pyautogui_controller import PyAutoGUIController

        return PyAutoGUIController(config)

    elif backend == "selenium":
        from .implementations.selenium_controller import SeleniumController

        return SeleniumController(config)

    elif backend == "native":
        return _create_native_input_controller(config)

    else:
        raise ValueError(f"Unsupported input controller backend: {backend}")


def _create_native_input_controller(config: HALConfig) -> IInputController:
    """Create native input controller for current platform.

    Args:
        config: HAL configuration

    Returns:
        Native input controller implementation

    Raises:
        ValueError: If no native implementation for platform
    """
    platform = _detect_platform(config)

    if platform == "windows":
        from .implementations.platform.windows_input import WindowsInputController

        return cast(IInputController, WindowsInputController(config))

    elif platform == "macos":
        from .implementations.platform.macos_input import MacOSInputController

        return cast(IInputController, MacOSInputController(config))

    elif platform == "linux":
        from .implementations.platform.linux_input import LinuxInputController

        return cast(IInputController, LinuxInputController(config))

    else:
        raise ValueError(f"No native input controller for platform: {platform}")


def _create_ocr_engine(config: HALConfig) -> IOCREngine:
    """Create OCR engine implementation.

    Args:
        config: HAL configuration

    Returns:
        OCR engine implementation

    Raises:
        ImportError: If backend is not available
        ValueError: If backend is not supported
    """
    backend = config.ocr_backend.lower()

    if backend == "easyocr":
        from .implementations.easyocr_engine import EasyOCREngine

        return EasyOCREngine(config)

    elif backend == "tesseract":
        from .implementations.tesseract_engine import TesseractEngine

        return TesseractEngine(config)

    elif backend == "cloud":
        from .implementations.cloud_ocr_engine import CloudOCREngine

        return CloudOCREngine(config)

    elif backend == "none":
        from .implementations.null_ocr_engine import NullOCREngine

        return NullOCREngine(config)

    else:
        raise ValueError(f"Unsupported OCR engine backend: {backend}")


def _create_platform_specific(config: HALConfig) -> IPlatformSpecific:
    """Create platform-specific implementation.

    Args:
        config: HAL configuration

    Returns:
        Platform-specific implementation

    Raises:
        ValueError: If platform is not supported
    """
    platform = _detect_platform(config)

    if platform == "windows":
        from .implementations.platform.windows import WindowsPlatform

        return WindowsPlatform(config)

    elif platform == "macos":
        from .implementations.platform.macos import MacOSPlatform

        return MacOSPlatform(config)

    elif platform == "linux":
        from .implementations.platform.linux import LinuxPlatform

        return LinuxPlatform(config)

    else:
        raise ValueError(f"Unsupported platform: {platform}")
