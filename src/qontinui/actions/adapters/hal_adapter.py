"""HAL-based adapter for action execution.

This adapter replaces PyAutoGUIAdapter and uses the HAL for all operations.
"""

from ...hal import HALFactory
from ...hal.interfaces import MouseButton as HALMouseButton
from ...logging import get_logger
from ..adapters import ActionAdapter, AdapterResult

logger = get_logger(__name__)


class HALAdapter(ActionAdapter):
    """HAL backend adapter with pure actions.

    This adapter uses the Hardware Abstraction Layer (HAL) for all
    GUI automation operations, providing better performance and
    flexibility compared to PyAutoGUI.
    """

    def __init__(self):
        """Initialize HAL adapter."""
        # Get HAL components
        self.screen_capture = HALFactory.get_screen_capture()
        self.input_controller = HALFactory.get_input_controller()
        self.pattern_matcher = HALFactory.get_pattern_matcher()
        self.ocr_engine = HALFactory.get_ocr_engine()

        logger.info("hal_adapter_initialized")

    # Mouse Actions

    def mouse_down(
        self, x: int | None = None, y: int | None = None, button: str = "left"
    ) -> AdapterResult:
        """Press and hold mouse button.

        Args:
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)
            button: Mouse button ('left', 'right', 'middle')

        Returns:
            AdapterResult with success status
        """
        try:
            # Convert button string to HAL enum
            hal_button = self._get_hal_button(button)

            # Execute mouse down
            success = self.input_controller.mouse_down(x, y, hal_button)

            if success:
                logger.debug(f"Mouse down at ({x}, {y}) with {button}")
                return AdapterResult(success=True, data=(x, y))
            else:
                return AdapterResult(success=False, error="Mouse down failed")

        except Exception as e:
            logger.error(f"Mouse down error: {e}")
            return AdapterResult(success=False, error=str(e))

    def mouse_up(
        self, x: int | None = None, y: int | None = None, button: str = "left"
    ) -> AdapterResult:
        """Release mouse button.

        Args:
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)
            button: Mouse button ('left', 'right', 'middle')

        Returns:
            AdapterResult with success status
        """
        try:
            # Convert button string to HAL enum
            hal_button = self._get_hal_button(button)

            # Execute mouse up
            success = self.input_controller.mouse_up(x, y, hal_button)

            if success:
                logger.debug(f"Mouse up at ({x}, {y}) with {button}")
                return AdapterResult(success=True, data=(x, y))
            else:
                return AdapterResult(success=False, error="Mouse up failed")

        except Exception as e:
            logger.error(f"Mouse up error: {e}")
            return AdapterResult(success=False, error=str(e))

    def mouse_move(self, x: int, y: int, duration: float = 0.0) -> AdapterResult:
        """Move mouse to coordinates.

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Movement duration in seconds

        Returns:
            AdapterResult with success status
        """
        try:
            # Execute mouse move
            success = self.input_controller.mouse_move(x, y, duration)

            if success:
                logger.debug(f"Mouse moved to ({x}, {y})")
                return AdapterResult(success=True, data=(x, y))
            else:
                return AdapterResult(success=False, error="Mouse move failed")

        except Exception as e:
            logger.error(f"Mouse move error: {e}")
            return AdapterResult(success=False, error=str(e))

    def mouse_click(self, x: int, y: int, button: str = "left") -> AdapterResult:
        """Single click at position.

        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button ('left', 'right', 'middle')

        Returns:
            AdapterResult with success status
        """
        try:
            # Convert button string to HAL enum
            hal_button = self._get_hal_button(button)

            # Execute click
            success = self.input_controller.mouse_click(x, y, hal_button)

            if success:
                logger.debug(f"Mouse clicked at ({x}, {y}) with {button}")
                return AdapterResult(success=True, data=(x, y))
            else:
                return AdapterResult(success=False, error="Mouse click failed")

        except Exception as e:
            logger.error(f"Mouse click error: {e}")
            return AdapterResult(success=False, error=str(e))

    def mouse_scroll(
        self, clicks: int, x: int | None = None, y: int | None = None
    ) -> AdapterResult:
        """Scroll mouse wheel.

        Args:
            clicks: Number of scroll clicks (positive=up, negative=down)
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)

        Returns:
            AdapterResult with success status
        """
        try:
            # Execute scroll
            success = self.input_controller.mouse_scroll(clicks, x, y)

            if success:
                logger.debug(f"Mouse scrolled {clicks} clicks")
                return AdapterResult(success=True, data=clicks)
            else:
                return AdapterResult(success=False, error="Mouse scroll failed")

        except Exception as e:
            logger.error(f"Mouse scroll error: {e}")
            return AdapterResult(success=False, error=str(e))

    # Keyboard Actions

    def key_down(self, key: str) -> AdapterResult:
        """Press and hold key.

        Args:
            key: Key to press down

        Returns:
            AdapterResult with success status
        """
        try:
            # Execute key down
            success = self.input_controller.key_down(key)

            if success:
                logger.debug(f"Key down: {key}")
                return AdapterResult(success=True, data=key)
            else:
                return AdapterResult(success=False, error="Key down failed")

        except Exception as e:
            logger.error(f"Key down error: {e}")
            return AdapterResult(success=False, error=str(e))

    def key_up(self, key: str) -> AdapterResult:
        """Release key.

        Args:
            key: Key to release

        Returns:
            AdapterResult with success status
        """
        try:
            # Execute key up
            success = self.input_controller.key_up(key)

            if success:
                logger.debug(f"Key up: {key}")
                return AdapterResult(success=True, data=key)
            else:
                return AdapterResult(success=False, error="Key up failed")

        except Exception as e:
            logger.error(f"Key up error: {e}")
            return AdapterResult(success=False, error=str(e))

    def key_press(self, key: str) -> AdapterResult:
        """Press key (down + up).

        Args:
            key: Key to press

        Returns:
            AdapterResult with success status
        """
        try:
            # Execute key press
            success = self.input_controller.key_press(key)

            if success:
                logger.debug(f"Key pressed: {key}")
                return AdapterResult(success=True, data=key)
            else:
                return AdapterResult(success=False, error="Key press failed")

        except Exception as e:
            logger.error(f"Key press error: {e}")
            return AdapterResult(success=False, error=str(e))

    def type_character(self, char: str) -> AdapterResult:
        """Type single character.

        Args:
            char: Character to type

        Returns:
            AdapterResult with success status
        """
        try:
            if len(char) != 1:
                return AdapterResult(success=False, error="Must be single character")

            # Type character
            success = self.input_controller.type_text(char)

            if success:
                logger.debug(f"Typed character: {char}")
                return AdapterResult(success=True, data=char)
            else:
                return AdapterResult(success=False, error="Type character failed")

        except Exception as e:
            logger.error(f"Type character error: {e}")
            return AdapterResult(success=False, error=str(e))

    # Screen Actions

    def capture_screen(self, region: tuple[int, int, int, int] | None = None) -> AdapterResult:
        """Capture screenshot.

        Args:
            region: Optional region (x, y, width, height)

        Returns:
            AdapterResult with screenshot image
        """
        try:
            if region:
                # Capture specific region
                x, y, width, height = region
                screenshot = self.screen_capture.capture_region(x, y, width, height)
            else:
                # Capture full screen
                screenshot = self.screen_capture.capture_screen()

            logger.debug(f"Screen captured: {screenshot.size}")
            return AdapterResult(success=True, data=screenshot)

        except Exception as e:
            logger.error(f"Screen capture error: {e}")
            return AdapterResult(success=False, error=str(e))

    def get_mouse_position(self) -> AdapterResult:
        """Get current mouse position.

        Returns:
            AdapterResult with position tuple (x, y)
        """
        try:
            pos = self.input_controller.get_mouse_position()

            logger.debug(f"Mouse position: ({pos.x}, {pos.y})")
            return AdapterResult(success=True, data=(pos.x, pos.y))

        except Exception as e:
            logger.error(f"Get mouse position error: {e}")
            return AdapterResult(success=False, error=str(e))

    def get_screen_size(self) -> AdapterResult:
        """Get screen dimensions.

        Returns:
            AdapterResult with size tuple (width, height)
        """
        try:
            primary = self.screen_capture.get_primary_monitor()

            if primary:
                size = (primary.width, primary.height)
                logger.debug(f"Screen size: {size}")
                return AdapterResult(success=True, data=size)
            else:
                return AdapterResult(success=False, error="No primary monitor found")

        except Exception as e:
            logger.error(f"Get screen size error: {e}")
            return AdapterResult(success=False, error=str(e))

    # Helper methods

    def _get_hal_button(self, button: str) -> HALMouseButton:
        """Convert button string to HAL MouseButton enum.

        Args:
            button: Button string ('left', 'right', 'middle')

        Returns:
            HAL MouseButton enum value
        """
        button_map = {
            "left": HALMouseButton.LEFT,
            "right": HALMouseButton.RIGHT,
            "middle": HALMouseButton.MIDDLE,
        }
        return button_map.get(button.lower(), HALMouseButton.LEFT)

    # Additional HAL-specific methods

    def find_pattern(self, screenshot, pattern, confidence: float = 0.9) -> AdapterResult:
        """Find pattern in screenshot using HAL pattern matcher.

        Args:
            screenshot: Screenshot image
            pattern: Pattern image to find
            confidence: Minimum confidence threshold

        Returns:
            AdapterResult with match location or None
        """
        try:
            match = self.pattern_matcher.find_pattern(screenshot, pattern, confidence)

            if match:
                logger.debug(f"Pattern found at ({match.x}, {match.y})")
                return AdapterResult(success=True, data=match)
            else:
                logger.debug("Pattern not found")
                return AdapterResult(success=True, data=None)

        except Exception as e:
            logger.error(f"Pattern matching error: {e}")
            return AdapterResult(success=False, error=str(e))

    def extract_text(self, image, languages: list | None = None) -> AdapterResult:
        """Extract text from image using HAL OCR engine.

        Args:
            image: Image to extract text from
            languages: List of language codes

        Returns:
            AdapterResult with extracted text
        """
        try:
            text = self.ocr_engine.extract_text(image, languages)

            logger.debug(f"Text extracted: {len(text)} characters")
            return AdapterResult(success=True, data=text)

        except Exception as e:
            logger.error(f"OCR error: {e}")
            return AdapterResult(success=False, error=str(e))
