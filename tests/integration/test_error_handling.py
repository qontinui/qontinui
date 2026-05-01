"""Integration tests for error handling.

Tests that new code raises appropriate exceptions, error messages are clear,
and recovery from errors works correctly.
"""

import threading
from unittest.mock import Mock

import pytest
from pynput import keyboard, mouse

from qontinui.actions.action_result import ActionResult
from qontinui.annotations.enhanced_state import state
from qontinui.annotations.state_registry import StateRegistry
from qontinui.exceptions import InputControlError
from qontinui.hal.implementations.keyboard_operations import KeyboardOperations
from qontinui.hal.implementations.mouse_operations import MouseOperations
from qontinui.hal.interfaces.input_controller import Key


class TestActionResultErrorHandling:
    """Test error handling in ActionResult."""

    def test_thread_safe_error_recovery(self):
        """Test that errors in one thread don't corrupt ActionResult."""
        result = ActionResult()
        errors: list[Exception] = []
        lock = threading.Lock()

        class MockMatch:
            def __init__(self, x: int):
                self.x = x

        def safe_worker():
            """Worker that adds matches normally."""
            try:
                for i in range(100):
                    match = MockMatch(i)
                    result.add(match)
            except Exception as e:
                with lock:
                    errors.append(e)

        def error_worker():
            """Worker that might cause errors."""
            try:
                for i in range(100):
                    # Simulate potential error
                    if i == 50:
                        # Try to add invalid data
                        try:
                            result.add(None)  # type: ignore
                        except (TypeError, AttributeError):
                            # Expected error - recover and continue
                            pass
                    else:
                        match = MockMatch(i)
                        result.add(match)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run mixed workers
        threads = [
            threading.Thread(target=safe_worker),
            threading.Thread(target=error_worker),
            threading.Thread(target=safe_worker),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify: should have matches from safe operations
        assert len(result.match_list) > 0
        # No unhandled errors
        assert len(errors) == 0

    def test_invalid_merge_operations(self):
        """Test error handling for invalid merge operations."""
        result = ActionResult()

        # Try to merge None
        with pytest.raises((TypeError, AttributeError)):
            result.add_all_results(None)  # type: ignore

        # Try to merge non-ActionResult
        with pytest.raises((TypeError, AttributeError)):
            result.add_all_results("invalid")  # type: ignore


class TestStateRegistryErrorHandling:
    """Test error handling in StateRegistry."""

    def test_register_invalid_state(self):
        """Test error when registering invalid state."""
        registry = StateRegistry()

        # Try to register non-decorated class
        class InvalidState:
            pass

        with pytest.raises((ValueError, AttributeError)):
            registry.register_state(InvalidState)

    def test_register_none_state(self):
        """Test error when registering None."""
        registry = StateRegistry()

        with pytest.raises((TypeError, ValueError, AttributeError)):
            registry.register_state(None)  # type: ignore

    def test_lookup_nonexistent_state(self):
        """Test lookup of nonexistent state returns None."""
        registry = StateRegistry()

        # Should return None, not error
        result = registry.get_state("nonexistent")
        assert result is None

        result_id = registry.get_state_id("nonexistent")
        assert result_id is None

    def test_concurrent_error_isolation(self):
        """Test that errors in one thread don't affect others."""
        registry = StateRegistry()
        errors: list[Exception] = []
        successes: list[int] = []
        lock = threading.Lock()

        def valid_worker(thread_id: int):
            """Worker that registers valid states."""
            try:
                for i in range(10):

                    @state(name=f"valid_{thread_id}_{i}")
                    class ValidState:
                        pass

                    registry.register_state(ValidState)

                with lock:
                    successes.append(thread_id)

            except Exception as e:
                with lock:
                    errors.append(e)

        def invalid_worker():
            """Worker that tries to register invalid states."""
            try:
                for _i in range(10):
                    # Try invalid operation
                    try:

                        class InvalidState:
                            pass

                        registry.register_state(InvalidState)
                    except (ValueError, AttributeError):
                        # Expected error - continue
                        pass

            except Exception as e:
                with lock:
                    errors.append(e)

        # Mix valid and invalid workers
        threads = [threading.Thread(target=valid_worker, args=(i,)) for i in range(5)]
        threads.append(threading.Thread(target=invalid_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify: valid operations succeeded despite invalid ones
        assert len(successes) == 5
        # Invalid worker didn't crash the registry
        assert len(registry.states) == 50  # 5 threads * 10 states


class TestKeyboardOperationsErrorHandling:
    """Test error handling in KeyboardOperations."""

    def test_keyboard_operation_failures(self):
        """Test that keyboard failures raise appropriate errors."""
        mock_keyboard = Mock(spec=keyboard.Controller)
        mock_keyboard.press.side_effect = OSError("Keyboard error")

        kbd_ops = KeyboardOperations(mock_keyboard)

        # Should raise InputControlError
        with pytest.raises(InputControlError) as exc_info:
            kbd_ops.key_press(Key.ENTER)

        # Error message should be informative
        error_msg = str(exc_info.value)
        assert "key_press" in error_msg.lower() or "keyboard" in error_msg.lower()

    def test_keyboard_error_cleanup(self):
        """Test that keyboard errors clean up properly."""
        mock_keyboard = Mock(spec=keyboard.Controller)

        # Simulate error during hotkey
        call_count = [0]

        def press_side_effect(key):
            call_count[0] += 1
            if call_count[0] == 2:  # Fail on second key
                raise OSError("Keyboard error")

        mock_keyboard.press.side_effect = press_side_effect

        kbd_ops = KeyboardOperations(mock_keyboard)

        # Hotkey should fail and attempt cleanup
        with pytest.raises(InputControlError):
            kbd_ops.hotkey(Key.CTRL, Key.SHIFT, "a")

        # Verify release was attempted (cleanup)
        assert mock_keyboard.release.called

    def test_keyboard_error_messages(self):
        """Test that error messages are clear."""
        mock_keyboard = Mock(spec=keyboard.Controller)
        mock_keyboard.type.side_effect = RuntimeError("Type failed")

        kbd_ops = KeyboardOperations(mock_keyboard)

        with pytest.raises(InputControlError) as exc_info:
            kbd_ops.type_text("test")

        # Error should mention the operation and original error
        error_str = str(exc_info.value)
        assert "type_text" in error_str.lower() or "type" in error_str.lower()


class TestMouseOperationsErrorHandling:
    """Test error handling in MouseOperations."""

    def test_mouse_operation_failures(self):
        """Test that mouse failures raise appropriate errors."""
        mock_mouse = Mock(spec=mouse.Controller)
        mock_mouse.position = (0, 0)
        mock_mouse.click.side_effect = OSError("Mouse error")

        mouse_ops = MouseOperations(mock_mouse)

        # Should raise InputControlError
        with pytest.raises(InputControlError) as exc_info:
            mouse_ops.mouse_click()

        # Error message should be informative
        error_msg = str(exc_info.value)
        assert "mouse" in error_msg.lower() or "click" in error_msg.lower()

    def test_mouse_drag_error_cleanup(self):
        """Test that mouse drag errors clean up properly."""
        mock_mouse = Mock(spec=mouse.Controller)
        mock_mouse.position = (0, 0)

        # Simulate error during drag
        call_count = [0]

        def position_setter(value):
            call_count[0] += 1
            if call_count[0] == 5:  # Fail during movement
                raise OSError("Mouse error")

        type(mock_mouse).position = property(lambda self: (0, 0), position_setter)

        mouse_ops = MouseOperations(mock_mouse)

        # Drag should fail and attempt cleanup
        with pytest.raises(InputControlError):
            mouse_ops.mouse_drag(0, 0, 100, 100)

        # Verify release was attempted (cleanup)
        assert mock_mouse.release.called

    def test_mouse_error_messages(self):
        """Test that error messages are clear."""
        mock_mouse = Mock(spec=mouse.Controller)
        mock_mouse.position = (0, 0)
        mock_mouse.scroll.side_effect = ValueError("Scroll failed")

        mouse_ops = MouseOperations(mock_mouse)

        with pytest.raises(InputControlError) as exc_info:
            mouse_ops.mouse_scroll(5)

        # Error should mention the operation
        error_str = str(exc_info.value)
        assert "scroll" in error_str.lower()


class TestErrorRecovery:
    """Test recovery mechanisms after errors."""

    def test_action_result_recovery_after_error(self):
        """Test ActionResult can continue after errors."""
        result = ActionResult()

        class MockMatch:
            def __init__(self, x: int):
                self.x = x

        # Add valid match
        result.add(MockMatch(1))

        # Try invalid operation
        try:
            result.add(None)  # type: ignore
        except (TypeError, AttributeError):
            pass

        # Should be able to continue
        result.add(MockMatch(2))

        assert len(result.match_list) == 2

    def test_state_registry_recovery_after_error(self):
        """Test StateRegistry can continue after errors."""
        registry = StateRegistry()

        # Register valid state
        @state(name="valid_1")
        class ValidState1:
            pass

        registry.register_state(ValidState1)

        # Try invalid registration
        try:

            class InvalidState:
                pass

            registry.register_state(InvalidState)
        except (ValueError, AttributeError):
            pass

        # Should be able to continue
        @state(name="valid_2")
        class ValidState2:
            pass

        registry.register_state(ValidState2)

        assert len(registry.states) == 2

    def test_concurrent_recovery(self):
        """Test recovery in concurrent scenarios."""
        result = ActionResult()
        errors: list[Exception] = []
        successes: list[int] = []
        lock = threading.Lock()

        class MockMatch:
            def __init__(self, x: int):
                self.x = x

        def worker_with_errors(worker_id: int):
            """Worker that encounters and recovers from errors."""
            try:
                success_count = 0
                for i in range(100):
                    try:
                        # Simulate occasional errors
                        if i % 25 == 0:
                            result.add(None)  # type: ignore
                        else:
                            match = MockMatch(worker_id * 100 + i)
                            result.add(match)
                            success_count += 1
                    except (TypeError, AttributeError):
                        # Recover from error
                        pass

                with lock:
                    successes.append(success_count)

            except Exception as e:
                with lock:
                    errors.append(e)

        # Run workers
        threads = [
            threading.Thread(target=worker_with_errors, args=(i,)) for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify recovery
        assert len(errors) == 0  # No unhandled errors
        assert len(successes) == 5
        assert sum(successes) == len(result.match_list)  # All successful adds recorded


class TestExceptionTypes:
    """Test that appropriate exception types are used."""

    def test_input_control_error_attributes(self):
        """Test InputControlError has proper attributes."""
        try:
            raise InputControlError("test_operation", "test error message")
        except InputControlError as e:
            # Should have informative string representation
            error_str = str(e)
            assert len(error_str) > 0
            assert "test" in error_str.lower()

    def test_exception_inheritance(self):
        """Test exception inheritance is correct."""
        # InputControlError should be an Exception
        assert issubclass(InputControlError, Exception)

        # Should be catchable as Exception
        try:
            raise InputControlError("test", "test")
        except Exception:
            pass  # Should catch

    def test_error_context_preservation(self):
        """Test that error context is preserved."""
        mock_keyboard = Mock(spec=keyboard.Controller)
        original_error = OSError("Original error message")
        mock_keyboard.press.side_effect = original_error

        kbd_ops = KeyboardOperations(mock_keyboard)

        try:
            kbd_ops.key_press(Key.ENTER)
        except InputControlError as e:
            # Original error should be preserved in chain
            assert e.__cause__ is original_error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
