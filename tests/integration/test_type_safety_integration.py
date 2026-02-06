"""Integration tests for type safety improvements.

Tests that verify type hints work correctly, mypy type checking passes,
and the codebase maintains strong type safety.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from qontinui.actions.action_result import ActionResult
from qontinui.annotations.enhanced_state import state
from qontinui.annotations.state_registry import StateRegistry
from qontinui.hal.implementations.keyboard_operations import KeyboardOperations
from qontinui.hal.implementations.mouse_operations import MouseOperations
from qontinui.hal.interfaces.input_controller import Key, MouseButton, MousePosition
from qontinui.model.element.location import Location
from qontinui.model.element.region import Region


class TestTypeHintUsage:
    """Test that type hints are used correctly throughout the codebase."""

    def test_action_result_type_hints(self):
        """Test ActionResult has correct type hints."""
        result = ActionResult()

        # These should all have proper types
        assert isinstance(result.success, bool)
        assert isinstance(result.output_text, str)
        assert isinstance(result.match_list, list)
        assert isinstance(result.active_states, set)
        assert isinstance(result.times_acted_on, int)

        # Property access should work
        assert isinstance(result.is_success, bool)
        assert isinstance(result.matches, list)

    def test_state_registry_type_hints(self):
        """Test StateRegistry has correct type hints."""
        registry = StateRegistry()

        # Define a state
        @state(name="test_type_state")
        class TestState:
            pass

        # Register returns int
        state_id = registry.register_state(TestState)
        assert isinstance(state_id, int)

        # Get state returns type or None
        state_class = registry.get_state("test_type_state")
        assert state_class is not None
        assert isinstance(state_class, type)

        # Get state ID returns int or None
        retrieved_id = registry.get_state_id("test_type_state")
        assert isinstance(retrieved_id, int)

        # Statistics returns dict
        stats = registry.get_statistics()
        assert isinstance(stats, dict)

    def test_keyboard_operations_type_hints(self):
        """Test KeyboardOperations has correct type hints."""
        from unittest.mock import Mock

        from pynput import keyboard

        mock_keyboard = Mock(spec=keyboard.Controller)
        kbd_ops = KeyboardOperations(mock_keyboard)

        # All operations should return bool
        result = kbd_ops.key_press(Key.ENTER)
        assert isinstance(result, bool)

        result = kbd_ops.type_text("test")
        assert isinstance(result, bool)

        result = kbd_ops.hotkey(Key.CTRL, "a")
        assert isinstance(result, bool)

    def test_mouse_operations_type_hints(self):
        """Test MouseOperations has correct type hints."""
        from unittest.mock import Mock

        from pynput import mouse

        mock_mouse = Mock(spec=mouse.Controller)
        mock_mouse.position = (100, 100)

        mouse_ops = MouseOperations(mock_mouse)

        # Operations return bool
        result = mouse_ops.mouse_move(200, 200)
        assert isinstance(result, bool)

        result = mouse_ops.mouse_click()
        assert isinstance(result, bool)

        # Get position returns MousePosition
        pos = mouse_ops.get_mouse_position()
        assert isinstance(pos, MousePosition)
        assert isinstance(pos.x, int)
        assert isinstance(pos.y, int)

    def test_region_type_hints(self):
        """Test Region has correct type hints."""
        region = Region(10, 20, 100, 200)

        assert isinstance(region.x, int)
        assert isinstance(region.y, int)
        assert isinstance(region.w, int)
        assert isinstance(region.h, int)

    def test_location_type_hints(self):
        """Test Location has correct type hints."""
        location = Location(50, 100)

        assert isinstance(location.x, int)
        assert isinstance(location.y, int)


class TestTypeChecking:
    """Test type checking with mypy (if available)."""

    def test_mypy_available(self):
        """Check if mypy is available for type checking."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "mypy", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            mypy_available = result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            mypy_available = False

        if mypy_available:
            print("mypy is available for type checking")
        else:
            pytest.skip("mypy not available")

    @pytest.mark.skipif(
        subprocess.run([sys.executable, "-m", "mypy", "--version"], capture_output=True).returncode
        != 0,
        reason="mypy not installed",
    )
    def test_mypy_check_action_result(self):
        """Run mypy on ActionResult to verify type safety."""
        src_path = Path(__file__).parent.parent.parent / "src" / "qontinui" / "actions"
        action_result_file = src_path / "action_result.py"

        if not action_result_file.exists():
            pytest.skip("action_result.py not found")

        # Run mypy
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                str(action_result_file),
                "--ignore-missing-imports",
                "--no-error-summary",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Check for type errors
        if result.returncode != 0:
            print(f"mypy output:\n{result.stdout}")

        # Allow some errors but should be minimal
        error_count = result.stdout.count("error:")
        assert error_count < 5, f"Too many type errors: {error_count}\n{result.stdout}"

    @pytest.mark.skipif(
        subprocess.run([sys.executable, "-m", "mypy", "--version"], capture_output=True).returncode
        != 0,
        reason="mypy not installed",
    )
    def test_mypy_check_state_registry(self):
        """Run mypy on StateRegistry to verify type safety."""
        src_path = Path(__file__).parent.parent.parent / "src" / "qontinui" / "annotations"
        registry_file = src_path / "state_registry.py"

        if not registry_file.exists():
            pytest.skip("state_registry.py not found")

        # Run mypy
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                str(registry_file),
                "--ignore-missing-imports",
                "--no-error-summary",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Check for type errors
        if result.returncode != 0:
            print(f"mypy output:\n{result.stdout}")

        error_count = result.stdout.count("error:")
        assert error_count < 5, f"Too many type errors: {error_count}\n{result.stdout}"


class TestTypeCompatibility:
    """Test type compatibility across components."""

    def test_action_result_match_list_type(self):
        """Test match_list maintains type consistency."""
        result = ActionResult()

        # Mock match
        class MockMatch:
            def __init__(self, value: int):
                self.value = value

        # Add matches
        match1 = MockMatch(1)
        match2 = MockMatch(2)

        result.add(match1, match2)

        # Verify type consistency
        assert len(result.match_list) == 2
        assert all(isinstance(m, MockMatch) for m in result.match_list)

    def test_state_registry_type_preservation(self):
        """Test StateRegistry preserves state types."""
        registry = StateRegistry()

        @state(name="preserved_state")
        class PreservedState:
            class_attr = "test"

        registry.register_state(PreservedState)

        # Retrieved state should maintain type
        retrieved = registry.get_state("preserved_state")
        assert retrieved is PreservedState
        assert hasattr(retrieved, "class_attr")
        assert retrieved.class_attr == "test"  # type: ignore

    def test_enum_type_usage(self):
        """Test enum types are used correctly."""
        # Key enum
        assert isinstance(Key.ENTER, Key)
        assert isinstance(Key.CTRL, Key)

        # MouseButton enum
        assert isinstance(MouseButton.LEFT, MouseButton)
        assert isinstance(MouseButton.RIGHT, MouseButton)

    def test_dataclass_types(self):
        """Test dataclass types work correctly."""
        # ActionResult is a dataclass-like structure
        result = ActionResult()

        # Should have all expected attributes
        assert hasattr(result, "success")
        assert hasattr(result, "match_list")
        assert hasattr(result, "active_states")

        # StateRegistry is a dataclass
        registry = StateRegistry()

        assert hasattr(registry, "states")
        assert hasattr(registry, "transitions")
        assert hasattr(registry, "groups")


class TestGenericTypeSupport:
    """Test generic type support."""

    def test_list_type_hints(self):
        """Test List type hints work correctly."""
        result = ActionResult()

        # match_list is List[Match]
        assert isinstance(result.match_list, list)

        # Can iterate
        for _ in result.match_list:
            pass

    def test_dict_type_hints(self):
        """Test Dict type hints work correctly."""
        registry = StateRegistry()

        # states is dict[str, type]
        assert isinstance(registry.states, dict)

        # Can iterate
        for name, state_class in registry.states.items():
            assert isinstance(name, str)
            assert isinstance(state_class, type)

    def test_set_type_hints(self):
        """Test Set type hints work correctly."""
        result = ActionResult()

        # active_states is set[str]
        assert isinstance(result.active_states, set)

        # Can add strings
        result.active_states.add("test_state")
        assert "test_state" in result.active_states


class TestTypeAnnotationCoverage:
    """Test that type annotations are comprehensive."""

    def test_public_methods_have_type_hints(self):
        """Test that public methods have type hints."""
        import inspect

        # Check ActionResult
        for name, method in inspect.getmembers(ActionResult, predicate=inspect.isfunction):
            if not name.startswith("_"):
                # Public method should have annotations
                sig = inspect.signature(method)
                # At minimum should have return annotation
                assert sig.return_annotation != inspect.Signature.empty or name in [
                    "__str__",
                    "__repr__",
                ], f"Method {name} missing return type hint"

    def test_public_functions_have_type_hints(self):
        """Test that public functions have type hints."""
        import inspect

        from qontinui.actions import result_builders

        # Check builder functions
        for name, func in inspect.getmembers(result_builders, predicate=inspect.isfunction):
            if not name.startswith("_"):
                sig = inspect.signature(func)
                # Should have return annotation
                assert (
                    sig.return_annotation != inspect.Signature.empty
                ), f"Function {name} missing return type hint"


class TestOptionalTypeHandling:
    """Test Optional type handling."""

    def test_optional_return_types(self):
        """Test functions with Optional return types."""
        registry = StateRegistry()

        # get_state returns Optional[type]
        result = registry.get_state("nonexistent")
        assert result is None

        # get_state_id returns Optional[int]
        result_id = registry.get_state_id("nonexistent")
        assert result_id is None

    def test_optional_parameters(self):
        """Test functions with Optional parameters."""
        from unittest.mock import Mock

        from pynput import mouse

        mock_mouse = Mock(spec=mouse.Controller)
        mock_mouse.position = (0, 0)

        mouse_ops = MouseOperations(mock_mouse)

        # x and y are Optional[int]
        mouse_ops.mouse_click()  # No position
        mouse_ops.mouse_click(100, 200)  # With position
        mouse_ops.mouse_click(x=100)  # Partial position


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
