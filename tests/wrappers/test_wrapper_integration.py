"""Integration tests for wrapper system."""

from PIL import Image

from qontinui.config.execution_mode import (
    ExecutionModeConfig,
    MockMode,
    reset_execution_mode,
    set_execution_mode,
)
from qontinui.model.element.region import Region
from qontinui.wrappers import get_controller
from qontinui.wrappers.controller import ExecutionModeController


class TestWrapperModeSwitch:
    """Test wrapper behavior when switching modes."""

    def setup_method(self):
        """Reset before each test."""
        reset_execution_mode()
        ExecutionModeController.reset_instance()

    def teardown_method(self):
        """Reset after each test."""
        reset_execution_mode()
        ExecutionModeController.reset_instance()

    def test_controller_singleton(self):
        """Test ExecutionModeController is singleton."""
        controller1 = get_controller()
        controller2 = get_controller()

        assert controller1 is controller2

    def test_controller_reset_instance(self):
        """Test controller reset creates new instance."""
        controller1 = get_controller()
        ExecutionModeController.reset_instance()
        controller2 = get_controller()

        assert controller1 is not controller2

    def test_mode_detection_in_mock_mode(self):
        """Test wrappers detect mock mode correctly."""
        set_execution_mode(ExecutionModeConfig(mode=MockMode.MOCK))

        controller = get_controller()

        assert controller.is_mock_mode()
        assert not controller.is_real_mode()

    def test_mode_detection_in_real_mode(self):
        """Test wrappers detect real mode correctly."""
        set_execution_mode(ExecutionModeConfig(mode=MockMode.REAL))

        controller = get_controller()

        assert not controller.is_mock_mode()
        assert controller.is_real_mode()

    def test_capture_wrapper_in_mock_mode(self):
        """Test CaptureWrapper uses MockCapture in mock mode."""
        set_execution_mode(ExecutionModeConfig(mode=MockMode.MOCK))

        controller = get_controller()
        screenshot = controller.capture.capture()

        # Should get a PIL Image from MockCapture
        assert isinstance(screenshot, Image.Image)
        # Mock capture generates 1920x1080 by default
        assert screenshot.width == 1920
        assert screenshot.height == 1080

    def test_capture_wrapper_region_in_mock_mode(self):
        """Test CaptureWrapper region capture in mock mode."""
        set_execution_mode(ExecutionModeConfig(mode=MockMode.MOCK))

        controller = get_controller()
        region = Region(100, 200, 300, 400)
        screenshot = controller.capture.capture_region(region)

        # Should get image matching region size
        assert isinstance(screenshot, Image.Image)
        assert screenshot.width == 300
        assert screenshot.height == 400

    def test_mouse_wrapper_in_mock_mode(self):
        """Test MouseWrapper uses MockMouse in mock mode."""
        set_execution_mode(ExecutionModeConfig(mode=MockMode.MOCK))

        controller = get_controller()
        result = controller.mouse.click(100, 200)

        # Click should succeed (return True)
        assert result is True

        # Verify operation was tracked
        mock_mouse = controller.mouse.mock_mouse
        assert mock_mouse.get_operation_count("click") == 1
        last_op = mock_mouse.get_last_operation()
        assert last_op["x"] == 100
        assert last_op["y"] == 200

    def test_keyboard_wrapper_in_mock_mode(self):
        """Test KeyboardWrapper uses MockKeyboard in mock mode."""
        set_execution_mode(ExecutionModeConfig(mode=MockMode.MOCK))

        controller = get_controller()
        result = controller.keyboard.type_text("Hello World")

        # Type should succeed
        assert result is True

        # Verify operation was tracked
        mock_keyboard = controller.keyboard.mock_keyboard
        typed_text = mock_keyboard.get_typed_text()
        assert typed_text == "Hello World"

    def test_time_wrapper_in_mock_mode(self):
        """Test TimeWrapper uses MockTime in mock mode (instant waits)."""
        set_execution_mode(ExecutionModeConfig(mode=MockMode.MOCK))

        controller = get_controller()

        # Get start time
        start = controller.time.now()

        # Wait 10 seconds (should be instant in mock mode)
        controller.time.wait(10.0)

        # Get end time
        end = controller.time.now()

        # Virtual time should have advanced 10 seconds
        elapsed = (end - start).total_seconds()
        assert elapsed == 10.0

    def test_mode_switching_affects_all_wrappers(self):
        """Test switching modes affects all wrappers."""
        controller = get_controller()

        # Start in mock mode
        controller.set_mock_mode()
        assert controller.is_mock_mode()
        assert controller.capture.is_mock_mode()
        assert controller.mouse.is_mock_mode()
        assert controller.keyboard.is_mock_mode()
        assert controller.time.is_mock_mode()

        # Switch to real mode
        controller.set_real_mode()
        assert controller.is_real_mode()
        assert controller.capture.is_real_mode()
        assert controller.mouse.is_real_mode()
        assert controller.keyboard.is_real_mode()
        assert controller.time.is_real_mode()

    def test_screenshot_mode_setup(self, tmp_path):
        """Test screenshot mode configuration."""
        screenshot_dir = tmp_path / "screenshots"
        screenshot_dir.mkdir()

        controller = get_controller()
        controller.set_screenshot_mode(str(screenshot_dir))

        mode = controller.get_execution_mode()
        assert mode.mode == MockMode.SCREENSHOT
        assert mode.screenshot_dir == str(screenshot_dir)


class TestWrapperOperationTracking:
    """Test wrapper operation tracking in mock mode."""

    def setup_method(self):
        """Setup mock mode for each test."""
        set_execution_mode(ExecutionModeConfig(mode=MockMode.MOCK))
        ExecutionModeController.reset_instance()

    def teardown_method(self):
        """Reset after each test."""
        reset_execution_mode()
        ExecutionModeController.reset_instance()

    def test_multiple_mouse_operations_tracked(self):
        """Test multiple mouse operations are tracked correctly."""
        controller = get_controller()

        controller.mouse.move(10, 20)
        controller.mouse.click(30, 40)
        controller.mouse.drag(50, 60, 100, 120)
        controller.mouse.scroll(5)

        mock_mouse = controller.mouse.mock_mouse
        assert mock_mouse.get_operation_count() == 4
        assert mock_mouse.get_operation_count("move") == 1
        assert mock_mouse.get_operation_count("click") == 1
        assert mock_mouse.get_operation_count("drag") == 1
        assert mock_mouse.get_operation_count("scroll") == 1

    def test_multiple_keyboard_operations_tracked(self):
        """Test multiple keyboard operations are tracked correctly."""
        controller = get_controller()

        controller.keyboard.type_text("Hello ")
        controller.keyboard.type_text("World")
        controller.keyboard.press("enter")
        controller.keyboard.hotkey("ctrl", "c")

        mock_keyboard = controller.keyboard.mock_keyboard
        assert mock_keyboard.get_operation_count() == 4
        assert mock_keyboard.get_operation_count("type_text") == 2
        assert mock_keyboard.get_operation_count("press") == 1
        assert mock_keyboard.get_operation_count("hotkey") == 1

    def test_virtual_time_accumulation(self):
        """Test virtual time accumulates across multiple waits."""
        controller = get_controller()

        start = controller.time.now()

        controller.time.wait(5.0)
        controller.time.wait(3.0)
        controller.time.wait(2.0)

        end = controller.time.now()
        elapsed = (end - start).total_seconds()

        assert elapsed == 10.0  # 5 + 3 + 2


class TestWrapperErrorHandling:
    """Test wrapper error handling."""

    def test_capture_wrapper_handles_errors_gracefully(self):
        """Test CaptureWrapper handles errors without crashing."""
        set_execution_mode(ExecutionModeConfig(mode=MockMode.MOCK))

        controller = get_controller()

        # Should handle invalid region gracefully
        # (MockCapture generates appropriate sized image)
        region = Region(-100, -200, 50, 75)
        screenshot = controller.capture.capture_region(region)

        # Should still return an image (even if coordinates are odd)
        assert isinstance(screenshot, Image.Image)


class TestWrapperPropertyAccess:
    """Test wrapper property access and lazy initialization."""

    def setup_method(self):
        """Reset controller."""
        ExecutionModeController.reset_instance()

    def teardown_method(self):
        """Reset controller."""
        ExecutionModeController.reset_instance()

    def test_lazy_wrapper_initialization(self):
        """Test wrappers are initialized lazily."""
        controller = get_controller()

        # Wrappers should be None before first access
        assert controller._find is None
        assert controller._capture is None
        assert controller._mouse is None
        assert controller._keyboard is None
        assert controller._time is None

        # Access each wrapper
        _ = controller.find
        _ = controller.capture
        _ = controller.mouse
        _ = controller.keyboard
        _ = controller.time

        # Now wrappers should be initialized
        assert controller._find is not None
        assert controller._capture is not None
        assert controller._mouse is not None
        assert controller._keyboard is not None
        assert controller._time is not None

    def test_wrapper_reuse(self):
        """Test same wrapper instance is reused."""
        controller = get_controller()

        wrapper1 = controller.capture
        wrapper2 = controller.capture

        assert wrapper1 is wrapper2
