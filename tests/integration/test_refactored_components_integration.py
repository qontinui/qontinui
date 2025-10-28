"""Integration tests for refactored KeyboardOperations and MouseOperations.

Tests that KeyboardOperations and MouseOperations work together correctly
with real input sequences and no interference between components.
"""

import threading
import time
from typing import List
from unittest.mock import MagicMock, Mock, call

import pytest
from pynput import keyboard, mouse

from qontinui.exceptions import InputControlError
from qontinui.hal.implementations.keyboard_operations import KeyboardOperations
from qontinui.hal.implementations.mouse_operations import MouseOperations
from qontinui.hal.interfaces.input_controller import Key, MouseButton, MousePosition


class TestKeyboardMouseIntegration:
    """Test KeyboardOperations and MouseOperations working together."""

    def test_keyboard_and_mouse_independent_operations(self):
        """Test that keyboard and mouse operations don't interfere."""
        # Create mock controllers
        mock_keyboard = Mock(spec=keyboard.Controller)
        mock_mouse = Mock(spec=mouse.Controller)
        mock_mouse.position = (100, 100)

        # Create operations
        kbd_ops = KeyboardOperations(mock_keyboard)
        mouse_ops = MouseOperations(mock_mouse)

        # Perform keyboard operations
        kbd_ops.type_text("Hello World")
        kbd_ops.key_press(Key.ENTER)

        # Perform mouse operations
        mouse_ops.mouse_move(500, 500)
        mouse_ops.mouse_click(button=MouseButton.LEFT)

        # Verify both worked independently
        assert mock_keyboard.type.called
        assert mock_keyboard.press.called
        assert mock_keyboard.release.called

        # Mouse operations should have been called
        assert mock_mouse.position == (500, 500)
        assert mock_mouse.click.called

    def test_coordinated_keyboard_mouse_workflow(self):
        """Test realistic workflow using both keyboard and mouse."""
        mock_keyboard = Mock(spec=keyboard.Controller)
        mock_mouse = Mock(spec=mouse.Controller)
        mock_mouse.position = (0, 0)

        kbd_ops = KeyboardOperations(mock_keyboard)
        mouse_ops = MouseOperations(mock_mouse)

        # Simulate: Click field, type text, press Enter
        mouse_ops.mouse_click(100, 50)  # Click text field
        time.sleep(0.01)  # Small delay
        kbd_ops.type_text("username@example.com")  # Type email
        kbd_ops.key_press(Key.TAB)  # Move to next field
        kbd_ops.type_text("password123")  # Type password
        kbd_ops.hotkey(Key.CTRL, Key.ENTER)  # Submit

        # Verify sequence
        mock_mouse.position = (100, 50)
        mock_mouse.click.assert_called()
        assert mock_keyboard.type.call_count >= 2  # Two text entries

    def test_concurrent_keyboard_mouse_operations(self):
        """Test keyboard and mouse operations executing concurrently."""
        mock_keyboard = Mock(spec=keyboard.Controller)
        mock_mouse = Mock(spec=mouse.Controller)
        mock_mouse.position = (0, 0)

        kbd_ops = KeyboardOperations(mock_keyboard)
        mouse_ops = MouseOperations(mock_mouse)

        errors: List[Exception] = []
        lock = threading.Lock()

        def keyboard_worker():
            """Perform keyboard operations."""
            try:
                for i in range(10):
                    kbd_ops.type_text(f"Text{i}")
                    time.sleep(0.01)
            except Exception as e:
                with lock:
                    errors.append(e)

        def mouse_worker():
            """Perform mouse operations."""
            try:
                for i in range(10):
                    mouse_ops.mouse_move(i * 10, i * 10)
                    time.sleep(0.01)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run concurrently
        threads = [threading.Thread(target=keyboard_worker), threading.Thread(target=mouse_worker)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"


class TestKeyboardOperationsIntegration:
    """Integration tests for KeyboardOperations."""

    def test_text_typing_with_special_keys(self):
        """Test typing text with special key combinations."""
        mock_keyboard = Mock(spec=keyboard.Controller)
        kbd_ops = KeyboardOperations(mock_keyboard)

        # Type text, then use special keys
        kbd_ops.type_text("Hello")
        kbd_ops.key_press(Key.SPACE)
        kbd_ops.type_text("World")
        kbd_ops.hotkey(Key.CTRL, "a")  # Select all
        kbd_ops.hotkey(Key.CTRL, "c")  # Copy

        # Verify operations
        assert mock_keyboard.type.call_count == 2
        assert mock_keyboard.press.call_count >= 4  # Space + Ctrl+a + Ctrl+c

    def test_keyboard_key_sequences(self):
        """Test sequences of key presses."""
        mock_keyboard = Mock(spec=keyboard.Controller)
        kbd_ops = KeyboardOperations(mock_keyboard)

        # Simulate navigation with arrow keys
        kbd_ops.key_press(Key.DOWN, presses=3)  # Down 3 times
        kbd_ops.key_press(Key.ENTER)  # Select
        kbd_ops.key_press(Key.TAB, presses=2)  # Tab twice

        # Verify sequence
        assert mock_keyboard.press.call_count >= 6  # 3 + 1 + 2

    def test_keyboard_hotkey_combinations(self):
        """Test various hotkey combinations."""
        mock_keyboard = Mock(spec=keyboard.Controller)
        kbd_ops = KeyboardOperations(mock_keyboard)

        # Test common hotkeys
        kbd_ops.hotkey(Key.CTRL, "s")  # Save
        kbd_ops.hotkey(Key.CTRL, Key.SHIFT, "s")  # Save As
        kbd_ops.hotkey(Key.ALT, Key.F4)  # Close

        # Verify all hotkeys were processed
        assert mock_keyboard.press.call_count >= 6  # At least 2 keys per hotkey

    def test_keyboard_error_handling(self):
        """Test keyboard error handling."""
        mock_keyboard = Mock(spec=keyboard.Controller)
        mock_keyboard.press.side_effect = OSError("Keyboard error")

        kbd_ops = KeyboardOperations(mock_keyboard)

        # Should raise InputControlError
        with pytest.raises(InputControlError) as exc_info:
            kbd_ops.key_press(Key.ENTER)

        assert "key_press" in str(exc_info.value)


class TestMouseOperationsIntegration:
    """Integration tests for MouseOperations."""

    def test_mouse_click_sequence(self):
        """Test sequence of mouse clicks."""
        mock_mouse = Mock(spec=mouse.Controller)
        mock_mouse.position = (0, 0)

        mouse_ops = MouseOperations(mock_mouse)

        # Click sequence: move, click, move, double-click
        mouse_ops.mouse_click(100, 100)
        mouse_ops.mouse_click(200, 200, clicks=2)  # Double click
        mouse_ops.mouse_click(300, 300, button=MouseButton.RIGHT)

        # Verify operations
        assert mock_mouse.position == (300, 300)
        assert mock_mouse.click.call_count >= 4  # 1 + 2 + 1

    def test_mouse_drag_operations(self):
        """Test mouse drag operations."""
        mock_mouse = Mock(spec=mouse.Controller)
        mock_mouse.position = (0, 0)

        mouse_ops = MouseOperations(mock_mouse)

        # Drag operation
        mouse_ops.mouse_drag(50, 50, 200, 200, duration=0.1)

        # Verify drag sequence: press -> move -> release
        assert mock_mouse.press.called
        assert mock_mouse.release.called

    def test_mouse_scroll_operations(self):
        """Test mouse scroll operations."""
        mock_mouse = Mock(spec=mouse.Controller)
        mock_mouse.position = (100, 100)

        mouse_ops = MouseOperations(mock_mouse)

        # Scroll operations
        mouse_ops.mouse_scroll(5)  # Scroll up 5 clicks
        mouse_ops.mouse_scroll(-3)  # Scroll down 3 clicks
        mouse_ops.mouse_scroll(10, x=200, y=200)  # Scroll at position

        # Verify scrolls
        assert mock_mouse.scroll.call_count == 3

    def test_mouse_position_tracking(self):
        """Test mouse position tracking."""
        mock_mouse = Mock(spec=mouse.Controller)
        mock_mouse.position = (100, 200)

        mouse_ops = MouseOperations(mock_mouse)

        # Get position
        pos = mouse_ops.get_mouse_position()

        assert isinstance(pos, MousePosition)
        assert pos.x == 100
        assert pos.y == 200

    def test_mouse_error_handling(self):
        """Test mouse error handling."""
        mock_mouse = Mock(spec=mouse.Controller)
        mock_mouse.position = (0, 0)
        mock_mouse.click.side_effect = OSError("Mouse error")

        mouse_ops = MouseOperations(mock_mouse)

        # Should raise InputControlError
        with pytest.raises(InputControlError) as exc_info:
            mouse_ops.mouse_click()

        assert "mouse_click" in str(exc_info.value)


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_form_filling_workflow(self):
        """Test complete form filling workflow."""
        mock_keyboard = Mock(spec=keyboard.Controller)
        mock_mouse = Mock(spec=mouse.Controller)
        mock_mouse.position = (0, 0)

        kbd_ops = KeyboardOperations(mock_keyboard)
        mouse_ops = MouseOperations(mock_mouse)

        # Form filling scenario
        # 1. Click first name field
        mouse_ops.mouse_click(100, 100)

        # 2. Type first name
        kbd_ops.type_text("John")

        # 3. Tab to last name
        kbd_ops.key_press(Key.TAB)

        # 4. Type last name
        kbd_ops.type_text("Doe")

        # 5. Tab to email
        kbd_ops.key_press(Key.TAB)

        # 6. Type email
        kbd_ops.type_text("john.doe@example.com")

        # 7. Click submit button
        mouse_ops.mouse_click(300, 400)

        # Verify workflow executed
        assert mock_keyboard.type.call_count == 3
        assert mock_keyboard.press.call_count >= 2  # Two tabs
        assert mock_mouse.click.call_count == 2  # Two clicks

    def test_text_editing_workflow(self):
        """Test text editing workflow with keyboard and mouse."""
        mock_keyboard = Mock(spec=keyboard.Controller)
        mock_mouse = Mock(spec=mouse.Controller)
        mock_mouse.position = (0, 0)

        kbd_ops = KeyboardOperations(mock_keyboard)
        mouse_ops = MouseOperations(mock_mouse)

        # Text editing scenario
        # 1. Click in text field
        mouse_ops.mouse_click(200, 150)

        # 2. Type some text
        kbd_ops.type_text("Initial text")

        # 3. Select all
        kbd_ops.hotkey(Key.CTRL, "a")

        # 4. Type replacement
        kbd_ops.type_text("New text")

        # 5. Save
        kbd_ops.hotkey(Key.CTRL, "s")

        # Verify
        assert mock_keyboard.type.call_count == 2
        assert mock_mouse.click.called

    def test_navigation_workflow(self):
        """Test navigation workflow with keyboard."""
        mock_keyboard = Mock(spec=keyboard.Controller)
        kbd_ops = KeyboardOperations(mock_keyboard)

        # Navigate through menu with keyboard
        kbd_ops.key_press(Key.DOWN)  # Move down
        kbd_ops.key_press(Key.DOWN)  # Move down again
        kbd_ops.key_press(Key.ENTER)  # Select
        kbd_ops.key_press(Key.TAB)  # Move to next element
        kbd_ops.key_press(Key.SPACE)  # Activate

        # Verify navigation
        assert mock_keyboard.press.call_count == 5

    def test_copy_paste_workflow(self):
        """Test copy-paste workflow."""
        mock_keyboard = Mock(spec=keyboard.Controller)
        mock_mouse = Mock(spec=mouse.Controller)
        mock_mouse.position = (0, 0)

        kbd_ops = KeyboardOperations(mock_keyboard)
        mouse_ops = MouseOperations(mock_mouse)

        # Copy from one location, paste to another
        # 1. Select text at first location
        mouse_ops.mouse_drag(100, 100, 200, 100)

        # 2. Copy
        kbd_ops.hotkey(Key.CTRL, "c")

        # 3. Click second location
        mouse_ops.mouse_click(100, 200)

        # 4. Paste
        kbd_ops.hotkey(Key.CTRL, "v")

        # Verify
        assert mock_mouse.press.called  # Drag operation
        assert mock_keyboard.press.call_count >= 4  # Ctrl+c and Ctrl+v


class TestConcurrentComponentUsage:
    """Test concurrent usage of keyboard and mouse components."""

    def test_multiple_threads_using_keyboard(self):
        """Multiple threads using keyboard operations."""
        mock_keyboard = Mock(spec=keyboard.Controller)
        kbd_ops = KeyboardOperations(mock_keyboard)

        errors: List[Exception] = []
        lock = threading.Lock()

        def type_worker(worker_id: int):
            """Worker that types text."""
            try:
                for i in range(5):
                    kbd_ops.type_text(f"Worker{worker_id}_Text{i}")
                    time.sleep(0.01)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run multiple keyboard workers
        threads = [threading.Thread(target=type_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0

    def test_multiple_threads_using_mouse(self):
        """Multiple threads using mouse operations."""
        mock_mouse = Mock(spec=mouse.Controller)
        mock_mouse.position = (0, 0)

        mouse_ops = MouseOperations(mock_mouse)

        errors: List[Exception] = []
        lock = threading.Lock()

        def click_worker(worker_id: int):
            """Worker that performs mouse clicks."""
            try:
                for i in range(5):
                    mouse_ops.mouse_click(worker_id * 50 + i * 10, i * 20)
                    time.sleep(0.01)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run multiple mouse workers
        threads = [threading.Thread(target=click_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
