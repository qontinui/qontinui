"""Tests for mock implementations."""

from datetime import datetime

from qontinui.mock.mock_capture import MockCapture
from qontinui.mock.mock_keyboard import MockKeyboard
from qontinui.mock.mock_mouse import MockMouse
from qontinui.mock.mock_time import MockTime
from qontinui.model.element.region import Region


class TestMockCapture:
    """Test MockCapture implementation."""

    def test_init(self):
        """Test MockCapture initialization."""
        capture = MockCapture()
        assert capture.screenshot_dir is None
        assert capture.mock_width == 1920
        assert capture.mock_height == 1080
        assert len(capture._monitors) == 1

    def test_init_with_screenshot_dir(self, tmp_path):
        """Test MockCapture with screenshot directory."""
        screenshot_dir = tmp_path / "screenshots"
        screenshot_dir.mkdir()

        capture = MockCapture(screenshot_dir=str(screenshot_dir))
        assert capture.screenshot_dir == screenshot_dir

    def test_capture_generates_mock_image(self):
        """Test capture generates mock image when no cache."""
        capture = MockCapture()
        screenshot = capture.capture()

        # Should generate mock image
        assert screenshot is not None
        assert screenshot.width == 1920
        assert screenshot.height == 1080

    def test_capture_region_generates_mock_image(self):
        """Test capture_region generates appropriate sized image."""
        capture = MockCapture()
        region = Region(100, 200, 300, 400)

        screenshot = capture.capture_region(region)

        # Should generate image matching region size
        assert screenshot is not None
        assert screenshot.width == 300
        assert screenshot.height == 400

    def test_get_monitors(self):
        """Test get_monitors returns mock monitor."""
        capture = MockCapture()
        monitors = capture.get_monitors()

        assert len(monitors) == 1
        assert monitors[0].width == 1920
        assert monitors[0].height == 1080
        assert monitors[0].is_primary

    def test_get_primary_monitor(self):
        """Test get_primary_monitor."""
        capture = MockCapture()
        primary = capture.get_primary_monitor()

        assert primary.is_primary
        assert primary.width == 1920

    def test_get_screen_size(self):
        """Test get_screen_size."""
        capture = MockCapture()
        width, height = capture.get_screen_size()

        assert width == 1920
        assert height == 1080

    def test_get_pixel_color_default(self):
        """Test get_pixel_color returns default color."""
        capture = MockCapture()
        color = capture.get_pixel_color(100, 200)

        # Should return mock color (light gray)
        assert color == (200, 200, 200)

    def test_save_screenshot(self, tmp_path):
        """Test save_screenshot."""
        capture = MockCapture()
        filepath = tmp_path / "test.png"

        result = capture.save_screenshot(str(filepath))

        assert result == str(filepath)
        assert filepath.exists()


class TestMockMouse:
    """Test MockMouse implementation."""

    def test_init(self):
        """Test MockMouse initialization."""
        mouse = MockMouse()
        assert mouse._x == 0
        assert mouse._y == 0
        assert len(mouse.operations) == 0

    def test_move(self):
        """Test mouse move tracking."""
        mouse = MockMouse()
        result = mouse.move(100, 200, duration=0.5)

        assert result is True
        assert mouse._x == 100
        assert mouse._y == 200
        assert len(mouse.operations) == 1
        assert mouse.operations[0]["type"] == "move"
        assert mouse.operations[0]["x"] == 100
        assert mouse.operations[0]["y"] == 200

    def test_click(self):
        """Test mouse click tracking."""
        mouse = MockMouse()
        result = mouse.click(100, 200, clicks=2)

        assert result is True
        assert mouse._x == 100
        assert mouse._y == 200
        assert len(mouse.operations) == 1
        assert mouse.operations[0]["type"] == "click"
        assert mouse.operations[0]["clicks"] == 2

    def test_click_current_position(self):
        """Test click at current position."""
        mouse = MockMouse()
        mouse.move(50, 75)
        mouse.click()  # No coordinates specified

        click_op = mouse.operations[1]
        assert click_op["x"] == 50
        assert click_op["y"] == 75

    def test_drag(self):
        """Test mouse drag tracking."""
        mouse = MockMouse()
        result = mouse.drag(10, 20, 100, 200, duration=1.0)

        assert result is True
        assert mouse._x == 100
        assert mouse._y == 200
        assert len(mouse.operations) == 1
        assert mouse.operations[0]["type"] == "drag"
        assert mouse.operations[0]["start_x"] == 10
        assert mouse.operations[0]["end_x"] == 100

    def test_scroll(self):
        """Test mouse scroll tracking."""
        mouse = MockMouse()
        result = mouse.scroll(5)

        assert result is True
        assert len(mouse.operations) == 1
        assert mouse.operations[0]["type"] == "scroll"
        assert mouse.operations[0]["clicks"] == 5

    def test_get_position(self):
        """Test get_position returns tracked position."""
        mouse = MockMouse()
        mouse.move(123, 456)

        position = mouse.get_position()
        assert position.x == 123
        assert position.y == 456

    def test_reset(self):
        """Test reset clears state."""
        mouse = MockMouse()
        mouse.move(100, 200)
        mouse.click()

        assert len(mouse.operations) == 2

        mouse.reset()

        assert mouse._x == 0
        assert mouse._y == 0
        assert len(mouse.operations) == 0

    def test_get_last_operation(self):
        """Test get_last_operation."""
        mouse = MockMouse()
        mouse.move(10, 20)
        mouse.click(30, 40)

        last_op = mouse.get_last_operation()
        assert last_op["type"] == "click"
        assert last_op["x"] == 30

    def test_get_operation_count(self):
        """Test get_operation_count."""
        mouse = MockMouse()
        mouse.click(10, 20)
        mouse.click(30, 40)
        mouse.move(50, 60)

        assert mouse.get_operation_count() == 3
        assert mouse.get_operation_count("click") == 2
        assert mouse.get_operation_count("move") == 1
        assert mouse.get_operation_count("scroll") == 0


class TestMockKeyboard:
    """Test MockKeyboard implementation."""

    def test_init(self):
        """Test MockKeyboard initialization."""
        keyboard = MockKeyboard()
        assert len(keyboard.operations) == 0
        assert len(keyboard.pressed_keys) == 0

    def test_type_text(self):
        """Test type_text tracking."""
        keyboard = MockKeyboard()
        result = keyboard.type_text("Hello World", interval=0.1)

        assert result is True
        assert len(keyboard.operations) == 1
        assert keyboard.operations[0]["type"] == "type_text"
        assert keyboard.operations[0]["text"] == "Hello World"

    def test_press(self):
        """Test key press tracking."""
        keyboard = MockKeyboard()
        result = keyboard.press("enter", presses=2)

        assert result is True
        assert len(keyboard.operations) == 1
        assert keyboard.operations[0]["type"] == "press"
        assert keyboard.operations[0]["key"] == "enter"
        assert keyboard.operations[0]["presses"] == 2

    def test_hotkey(self):
        """Test hotkey tracking."""
        keyboard = MockKeyboard()
        result = keyboard.hotkey("ctrl", "c")

        assert result is True
        assert len(keyboard.operations) == 1
        assert keyboard.operations[0]["type"] == "hotkey"
        assert keyboard.operations[0]["keys"] == ["ctrl", "c"]

    def test_key_down_up(self):
        """Test key down/up tracking."""
        keyboard = MockKeyboard()

        keyboard.key_down("shift")
        assert "shift" in keyboard.pressed_keys

        keyboard.key_up("shift")
        assert "shift" not in keyboard.pressed_keys

    def test_is_key_pressed(self):
        """Test is_key_pressed."""
        keyboard = MockKeyboard()

        assert not keyboard.is_key_pressed("ctrl")

        keyboard.key_down("ctrl")
        assert keyboard.is_key_pressed("ctrl")

        keyboard.key_up("ctrl")
        assert not keyboard.is_key_pressed("ctrl")

    def test_reset(self):
        """Test reset clears state."""
        keyboard = MockKeyboard()
        keyboard.type_text("test")
        keyboard.key_down("shift")

        keyboard.reset()

        assert len(keyboard.operations) == 0
        assert len(keyboard.pressed_keys) == 0

    def test_get_last_operation(self):
        """Test get_last_operation."""
        keyboard = MockKeyboard()
        keyboard.press("a")
        keyboard.type_text("test")

        last_op = keyboard.get_last_operation()
        assert last_op["type"] == "type_text"
        assert last_op["text"] == "test"

    def test_get_operation_count(self):
        """Test get_operation_count."""
        keyboard = MockKeyboard()
        keyboard.type_text("Hello")
        keyboard.type_text("World")
        keyboard.press("enter")

        assert keyboard.get_operation_count() == 3
        assert keyboard.get_operation_count("type_text") == 2
        assert keyboard.get_operation_count("press") == 1

    def test_get_typed_text(self):
        """Test get_typed_text."""
        keyboard = MockKeyboard()
        keyboard.type_text("Hello ")
        keyboard.press("enter")
        keyboard.type_text("World")

        typed = keyboard.get_typed_text()
        assert typed == "Hello World"


class TestMockTime:
    """Test MockTime implementation."""

    def test_init_instant_mode(self):
        """Test MockTime initialization in instant mode."""
        mock_time = MockTime(instant_mode=True)

        assert mock_time.instant_mode is True
        assert mock_time.time_scale == 0.0
        assert isinstance(mock_time.virtual_time, datetime)

    def test_init_scaled_mode(self):
        """Test MockTime initialization in scaled mode."""
        mock_time = MockTime(instant_mode=False, time_scale=0.1)

        assert mock_time.instant_mode is False
        assert mock_time.time_scale == 0.1

    def test_wait_instant_mode(self):
        """Test wait in instant mode advances virtual time."""
        mock_time = MockTime(instant_mode=True)
        start_time = mock_time.virtual_time

        mock_time.wait(5.0)

        elapsed = mock_time.virtual_time - start_time
        assert elapsed.total_seconds() == 5.0

    def test_now(self):
        """Test now returns virtual time."""
        mock_time = MockTime()
        mock_time.wait(10.0)

        now = mock_time.now()
        elapsed = now - mock_time.start_time

        assert elapsed.total_seconds() >= 10.0

    def test_timestamp(self):
        """Test timestamp returns virtual timestamp."""
        mock_time = MockTime()
        ts1 = mock_time.timestamp()

        mock_time.wait(5.0)
        ts2 = mock_time.timestamp()

        assert ts2 - ts1 >= 5.0

    def test_wait_until_success(self):
        """Test wait_until when condition becomes true."""
        mock_time = MockTime()
        counter = {"value": 0}

        def condition():
            counter["value"] += 1
            return counter["value"] >= 3

        result = mock_time.wait_until(condition, timeout=10.0, poll_interval=0.5)

        assert result is True
        assert counter["value"] >= 3

    def test_wait_until_timeout(self):
        """Test wait_until when condition never becomes true."""
        mock_time = MockTime()

        def condition():
            return False

        result = mock_time.wait_until(condition, timeout=2.0, poll_interval=0.5)

        assert result is False

    def test_measure(self):
        """Test measure tracks virtual time."""
        mock_time = MockTime()

        def test_func():
            mock_time.wait(3.0)
            return "result"

        result, duration = mock_time.measure(test_func)

        assert result == "result"
        assert duration >= 3.0

    def test_set_time(self):
        """Test set_time."""
        mock_time = MockTime()
        new_time = datetime(2025, 1, 1, 12, 0, 0)

        mock_time.set_time(new_time)

        assert mock_time.virtual_time == new_time

    def test_advance(self):
        """Test advance."""
        mock_time = MockTime()
        start_time = mock_time.virtual_time

        mock_time.advance(3600)  # 1 hour

        elapsed = mock_time.virtual_time - start_time
        assert elapsed.total_seconds() == 3600

    def test_reset(self):
        """Test reset."""
        mock_time = MockTime()
        mock_time.wait(100)

        mock_time.reset()

        assert mock_time.virtual_time == mock_time.start_time

    def test_get_elapsed_time(self):
        """Test get_elapsed_time."""
        mock_time = MockTime()
        mock_time.wait(10.0)

        elapsed = mock_time.get_elapsed_time()
        assert elapsed.total_seconds() == 10.0

    def test_set_instant_mode(self):
        """Test set_instant_mode."""
        mock_time = MockTime(instant_mode=False)
        assert not mock_time.instant_mode

        mock_time.set_instant_mode(True)
        assert mock_time.instant_mode

    def test_set_time_scale(self):
        """Test set_time_scale."""
        mock_time = MockTime()
        mock_time.set_time_scale(0.5)

        assert mock_time.time_scale == 0.5
