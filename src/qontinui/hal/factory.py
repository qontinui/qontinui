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
from typing import Any, cast

from .config import HALConfig, get_config
from .interfaces import (
    IOCREngine,
    IPatternMatcher,
    IPlatformSpecific,
    IScreenCapture,
)
from .interfaces.keyboard_controller import IKeyboardController
from .interfaces.mouse_controller import IMouseController


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
    def get_input_controller(cls, config: HALConfig | None = None) -> Any:
        """Get unified input controller (keyboard + mouse).

        This method provides backward compatibility for code that expects
        a unified input controller. Returns an object with both keyboard
        and mouse controller methods.

        Args:
            config: Optional configuration override

        Returns:
            Object with keyboard and mouse controller methods

        Note:
            This is a compatibility shim. New code should use HALContainer
            and access keyboard_controller and mouse_controller separately.
        """
        from types import SimpleNamespace

        config = config or get_config()
        keyboard = cls.get_keyboard_controller(config)
        mouse = cls.get_mouse_controller(config)

        # Create a combined controller object
        return SimpleNamespace(
            keyboard=keyboard,
            mouse=mouse,
            # Expose keyboard methods at top level for convenience
            press=keyboard.key_press,
            key_press=keyboard.key_press,
            release=keyboard.key_up,
            key_up=keyboard.key_up,
            key_down=keyboard.key_down,
            type=keyboard.type_text,
            type_text=keyboard.type_text,
            hotkey=keyboard.hotkey,
            # Expose mouse methods at top level for convenience
            move=mouse.mouse_move,
            mouse_move=mouse.mouse_move,
            move_relative=mouse.mouse_move_relative,
            mouse_move_relative=mouse.mouse_move_relative,
            click=mouse.mouse_click,
            mouse_click=mouse.mouse_click,
            mouse_down=mouse.mouse_down,
            mouse_up=mouse.mouse_up,
            mouse_drag=mouse.mouse_drag,
            drag=mouse.drag,
            click_at=mouse.click_at,
            double_click_at=mouse.double_click_at,
            scroll=mouse.mouse_scroll,
            mouse_scroll=mouse.mouse_scroll,
            get_position=mouse.get_mouse_position,
            get_mouse_position=mouse.get_mouse_position,
            move_mouse=mouse.move_mouse,
        )

    @classmethod
    def get_keyboard_controller(cls, config: HALConfig | None = None) -> IKeyboardController:
        """Get keyboard controller implementation.

        Args:
            config: Optional configuration override

        Returns:
            IKeyboardController implementation
        """
        config = config or get_config()
        cache_key = f"keyboard_controller_{config.input_backend}"

        # Use lock to prevent concurrent imports that could cause deadlock
        with cls._lock:
            if cache_key not in cls._instances:
                backend = config.input_backend.lower()

                if backend == "pynput":
                    from pynput import keyboard

                    from .implementations.keyboard_operations import KeyboardOperations

                    cls._instances[cache_key] = KeyboardOperations(keyboard.Controller())
                elif backend == "pyautogui":
                    raise NotImplementedError("PyAutoGUI keyboard backend not yet implemented")
                elif backend == "selenium":
                    raise NotImplementedError("Selenium keyboard backend not yet implemented")
                elif backend == "native":
                    cls._instances[cache_key] = cls._get_native_keyboard_controller(config)
                else:
                    raise ValueError(f"Unsupported keyboard controller backend: {backend}")

            instance: IKeyboardController = cls._instances[cache_key]
            return instance

    @classmethod
    def get_mouse_controller(cls, config: HALConfig | None = None) -> IMouseController:
        """Get mouse controller implementation.

        Args:
            config: Optional configuration override

        Returns:
            IMouseController implementation
        """
        config = config or get_config()
        cache_key = f"mouse_controller_{config.input_backend}"

        # Use lock to prevent concurrent imports that could cause deadlock
        with cls._lock:
            if cache_key not in cls._instances:
                backend = config.input_backend.lower()

                if backend == "pynput":
                    from pynput import mouse

                    from .implementations.mouse_operations import MouseOperations

                    cls._instances[cache_key] = MouseOperations(mouse.Controller())
                elif backend == "pyautogui":
                    raise NotImplementedError("PyAutoGUI mouse backend not yet implemented")
                elif backend == "selenium":
                    raise NotImplementedError("Selenium mouse backend not yet implemented")
                elif backend == "native":
                    cls._instances[cache_key] = cls._get_native_mouse_controller(config)
                else:
                    raise ValueError(f"Unsupported mouse controller backend: {backend}")

            instance: IMouseController = cls._instances[cache_key]
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
    def _get_native_keyboard_controller(cls, config: HALConfig) -> IKeyboardController:
        """Get native keyboard controller for current platform.

        Args:
            config: HAL configuration

        Returns:
            Native keyboard controller implementation
        """
        # For now, use pynput as the native keyboard controller on all platforms
        from pynput import keyboard

        from .implementations.keyboard_operations import KeyboardOperations

        return KeyboardOperations(keyboard.Controller())

    @classmethod
    def _get_native_mouse_controller(cls, config: HALConfig) -> IMouseController:
        """Get native mouse controller for current platform.

        Args:
            config: HAL configuration

        Returns:
            Native mouse controller implementation
        """
        # For now, use pynput as the native mouse controller on all platforms
        from pynput import mouse

        from .implementations.mouse_operations import MouseOperations

        return MouseOperations(mouse.Controller())

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
