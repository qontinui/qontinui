"""HAL Container for dependency injection.

This module provides a container for HAL component instances, replacing
the global factory pattern with explicit dependency injection.
"""

from dataclasses import dataclass, field

from .config import HALConfig
from .interfaces import (
    IAccessibilityCapture,
    IOCREngine,
    IPatternMatcher,
    IPlatformSpecific,
    IScreenCapture,
)
from .interfaces.keyboard_controller import IKeyboardController
from .interfaces.mouse_controller import IMouseController


@dataclass
class HALContainer:
    """Container for all HAL component instances.

    This container holds references to all HAL implementation instances
    and provides them to components that need them. It eliminates the
    need for global state and factory methods.

    Components are created once at application startup and shared across
    the application lifetime.

    BREAKING CHANGE: input_controller has been split into keyboard_controller
    and mouse_controller for better separation of concerns.

    Attributes:
        keyboard_controller: Keyboard input control implementation
        mouse_controller: Mouse input control implementation
        screen_capture: Screen capture implementation
        pattern_matcher: Image pattern matching implementation
        ocr_engine: OCR text recognition implementation
        platform_specific: Platform-specific utilities implementation
        accessibility_capture: Optional accessibility tree capture implementation
        config: HAL configuration used to create these components

    Example:
        >>> from qontinui.hal import HALConfig, initialize_hal
        >>> config = HALConfig()
        >>> hal = initialize_hal(config)
        >>> # Use separate controllers
        >>> hal.keyboard_controller.type_text("Hello")
        >>> hal.mouse_controller.mouse_move(100, 200)
    """

    keyboard_controller: IKeyboardController
    mouse_controller: IMouseController
    screen_capture: IScreenCapture
    pattern_matcher: IPatternMatcher
    ocr_engine: IOCREngine
    platform_specific: IPlatformSpecific
    config: HALConfig
    accessibility_capture: IAccessibilityCapture | None = field(default=None)

    @classmethod
    def create_from_config(cls, config: HALConfig) -> "HALContainer":
        """Create HAL container from configuration.

        This method eagerly initializes all HAL components based on the
        provided configuration. Initialization happens synchronously and
        any errors are raised immediately.

        Args:
            config: HAL configuration specifying which backends to use

        Returns:
            HALContainer with all components initialized

        Raises:
            ImportError: If required backend library is not installed
            ValueError: If backend configuration is invalid
            RuntimeError: If component initialization fails

        Example:
            >>> config = HALConfig(
            ...     input_backend="pynput",
            ...     capture_backend="mss",
            ...     matcher_backend="opencv"
            ... )
            >>> container = HALContainer.create_from_config(config)
        """
        # Import the initialization module
        from .initialization import (
            _create_accessibility_capture,
            _create_keyboard_controller,
            _create_mouse_controller,
            _create_ocr_engine,
            _create_pattern_matcher,
            _create_platform_specific,
            _create_screen_capture,
        )

        # Create all components eagerly
        keyboard_controller = _create_keyboard_controller(config)
        mouse_controller = _create_mouse_controller(config)
        screen_capture = _create_screen_capture(config)
        pattern_matcher = _create_pattern_matcher(config)
        ocr_engine = _create_ocr_engine(config)
        platform_specific = _create_platform_specific(config)
        accessibility_capture = _create_accessibility_capture(config)

        return cls(
            keyboard_controller=keyboard_controller,
            mouse_controller=mouse_controller,
            screen_capture=screen_capture,
            pattern_matcher=pattern_matcher,
            ocr_engine=ocr_engine,
            platform_specific=platform_specific,
            config=config,
            accessibility_capture=accessibility_capture,
        )

    def cleanup(self) -> None:
        """Clean up HAL component resources.

        This method should be called when the application is shutting down
        to properly release resources held by HAL components.

        Example:
            >>> hal = initialize_hal(config)
            >>> try:
            ...     # Use hal
            ...     pass
            ... finally:
            ...     hal.cleanup()
        """
        # Clean up accessibility capture if present (it has async disconnect)
        # Note: For proper async cleanup, use shutdown_hal_async() instead
        if self.accessibility_capture is not None:
            # Accessibility capture has async disconnect - we can't await here
            # The caller should handle async cleanup separately if needed
            pass

    async def cleanup_async(self) -> None:
        """Async cleanup for components that need it.

        Use this method when you need to properly clean up async components
        like accessibility capture.

        Example:
            >>> hal = initialize_hal(config)
            >>> try:
            ...     # Use hal
            ...     pass
            ... finally:
            ...     await hal.cleanup_async()
        """
        if self.accessibility_capture is not None:
            await self.accessibility_capture.disconnect()
