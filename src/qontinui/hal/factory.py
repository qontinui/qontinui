"""HAL Factory for creating implementation instances.

DEPRECATED: This factory pattern is deprecated in favor of explicit dependency
injection using HALContainer. The global singleton pattern causes circular
dependencies and makes testing difficult.

Migration Guide:
    Old (deprecated):
        >>> from qontinui.hal import HALFactory
        >>> controller = HALFactory.get_input_controller()

    New (recommended):
        >>> from qontinui.hal import initialize_hal
        >>> hal = initialize_hal()
        >>> controller = hal.input_controller

    For ActionExecutor:
        Old:
            >>> executor = ActionExecutor(config)
            >>> # Uses HALFactory internally

        New:
            >>> hal = initialize_hal(config)
            >>> executor = ActionExecutor(config, hal=hal)

See HALContainer and initialize_hal() for the new dependency injection pattern.
"""

import sys
import threading
import warnings
from typing import Any, cast

from .config import HALConfig, get_config
from .interfaces import (
    IInputController,
    IOCREngine,
    IPatternMatcher,
    IPlatformSpecific,
    IScreenCapture,
)


class HALFactory:
    """Factory for creating HAL implementation instances.

    This factory manages the creation and caching of HAL implementations
    based on configuration settings.
    """

    # Cached instances
    _instances: dict[str, Any] = {}
    _lock = threading.Lock()

    @classmethod
    def get_screen_capture(cls, config: HALConfig | None = None) -> IScreenCapture:
        """Get screen capture implementation.

        Args:
            config: Optional configuration override

        Returns:
            IScreenCapture implementation

        Raises:
            ImportError: If backend is not available
            ValueError: If backend is not supported
        """
        config = config or get_config()
        cache_key = f"screen_capture_{config.capture_backend}"

        if cache_key not in cls._instances:
            backend = config.capture_backend.lower()

            if backend == "mss":
                from .implementations.mss_capture import MSSScreenCapture

                cls._instances[cache_key] = MSSScreenCapture(config)
            elif backend == "pyautogui":
                from .implementations.pyautogui_capture import PyAutoGUIScreenCapture

                cls._instances[cache_key] = PyAutoGUIScreenCapture(config)
            elif backend == "pillow":
                from .implementations.pillow_capture import PillowScreenCapture

                cls._instances[cache_key] = PillowScreenCapture(config)
            elif backend == "native":
                cls._instances[cache_key] = cls._get_native_screen_capture(config)
            else:
                raise ValueError(f"Unsupported screen capture backend: {backend}")

        instance: IScreenCapture = cls._instances[cache_key]
        return instance

    @classmethod
    def get_pattern_matcher(cls, config: HALConfig | None = None) -> IPatternMatcher:
        """Get pattern matcher implementation.

        Args:
            config: Optional configuration override

        Returns:
            IPatternMatcher implementation
        """
        config = config or get_config()
        cache_key = f"pattern_matcher_{config.matcher_backend}"

        if cache_key not in cls._instances:
            backend = config.matcher_backend.lower()

            if backend == "opencv":
                from .implementations.opencv_matcher import OpenCVMatcher

                cls._instances[cache_key] = OpenCVMatcher(config)
            elif backend == "pyautogui":
                from .implementations.pyautogui_matcher import PyAutoGUIMatcher

                cls._instances[cache_key] = PyAutoGUIMatcher(config)
            elif backend == "tensorflow":
                from .implementations.tensorflow_matcher import TensorFlowMatcher

                cls._instances[cache_key] = TensorFlowMatcher(config)
            elif backend == "native":
                cls._instances[cache_key] = cls._get_native_pattern_matcher(config)
            else:
                raise ValueError(f"Unsupported pattern matcher backend: {backend}")

        instance: IPatternMatcher = cls._instances[cache_key]
        return instance

    @classmethod
    def get_input_controller(cls, config: HALConfig | None = None) -> IInputController:
        """Get input controller implementation.

        Args:
            config: Optional configuration override

        Returns:
            IInputController implementation
        """
        config = config or get_config()
        cache_key = f"input_controller_{config.input_backend}"

        # Use lock to prevent concurrent imports that could cause deadlock
        with cls._lock:
            if cache_key not in cls._instances:
                backend = config.input_backend.lower()

                if backend == "pynput":
                    from .implementations.pynput_controller import PynputController

                    cls._instances[cache_key] = PynputController(config)
                elif backend == "pyautogui":
                    from .implementations.pyautogui_controller import PyAutoGUIController

                    cls._instances[cache_key] = PyAutoGUIController(config)
                elif backend == "selenium":
                    from .implementations.selenium_controller import SeleniumController

                    cls._instances[cache_key] = SeleniumController(config)
                elif backend == "native":
                    cls._instances[cache_key] = cls._get_native_input_controller(config)
                else:
                    raise ValueError(f"Unsupported input controller backend: {backend}")

            instance: IInputController = cls._instances[cache_key]
            return instance

    @classmethod
    def get_ocr_engine(cls, config: HALConfig | None = None) -> IOCREngine:
        """Get OCR engine implementation.

        Args:
            config: Optional configuration override

        Returns:
            IOCREngine implementation
        """
        config = config or get_config()
        cache_key = f"ocr_engine_{config.ocr_backend}"

        if cache_key not in cls._instances:
            backend = config.ocr_backend.lower()

            if backend == "easyocr":
                from .implementations.easyocr_engine import EasyOCREngine

                cls._instances[cache_key] = EasyOCREngine(config)
            elif backend == "tesseract":
                from .implementations.tesseract_engine import TesseractEngine

                cls._instances[cache_key] = TesseractEngine(config)
            elif backend == "cloud":
                from .implementations.cloud_ocr_engine import CloudOCREngine

                cls._instances[cache_key] = CloudOCREngine(config)
            elif backend == "none":
                from .implementations.null_ocr_engine import NullOCREngine

                cls._instances[cache_key] = NullOCREngine(config)
            else:
                raise ValueError(f"Unsupported OCR engine backend: {backend}")

        instance: IOCREngine = cls._instances[cache_key]
        return instance

    @classmethod
    def get_platform_specific(cls, config: HALConfig | None = None) -> IPlatformSpecific:
        """Get platform-specific implementation.

        Args:
            config: Optional configuration override

        Returns:
            IPlatformSpecific implementation
        """
        config = config or get_config()
        cache_key = "platform_specific"

        if cache_key not in cls._instances:
            platform = cls._detect_platform(config)

            if platform == "windows":
                from .implementations.platform.windows import WindowsPlatform

                cls._instances[cache_key] = WindowsPlatform(config)
            elif platform == "macos":
                from .implementations.platform.macos import MacOSPlatform

                cls._instances[cache_key] = MacOSPlatform(config)
            elif platform == "linux":
                from .implementations.platform.linux import LinuxPlatform

                cls._instances[cache_key] = LinuxPlatform(config)
            else:
                raise ValueError(f"Unsupported platform: {platform}")

        instance: IPlatformSpecific = cls._instances[cache_key]
        return instance

    @classmethod
    def _detect_platform(cls, config: HALConfig) -> str:
        """Detect current platform.

        Args:
            config: HAL configuration

        Returns:
            Platform name ('windows', 'macos', 'linux')
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

    @classmethod
    def _get_native_screen_capture(cls, config: HALConfig) -> IScreenCapture:
        """Get native screen capture for current platform.

        Args:
            config: HAL configuration

        Returns:
            Native screen capture implementation
        """

        platform = cls._detect_platform(config)

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

    @classmethod
    def _get_native_pattern_matcher(cls, config: HALConfig) -> IPatternMatcher:
        """Get native pattern matcher for current platform.

        Args:
            config: HAL configuration

        Returns:
            Native pattern matcher implementation
        """
        # Most platforms will use OpenCV as "native"
        from .implementations.opencv_matcher import OpenCVMatcher

        return OpenCVMatcher(config)

    @classmethod
    def _get_native_input_controller(cls, config: HALConfig) -> IInputController:
        """Get native input controller for current platform.

        Args:
            config: HAL configuration

        Returns:
            Native input controller implementation
        """

        platform = cls._detect_platform(config)

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

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached instances."""
        cls._instances.clear()

    @classmethod
    def get_instance_count(cls) -> int:
        """Get number of cached instances.

        Returns:
            Number of cached instances
        """
        return len(cls._instances)
