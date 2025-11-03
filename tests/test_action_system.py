"""Test the Brobot-style action system."""

import pytest

from qontinui.actions import (
    ActionConfig,
)
from qontinui.actions.basic.click.click_options import ClickOptionsBuilder
from qontinui.actions.composite.drag.drag_options import DragOptionsBuilder
from qontinui.primitives import MouseClick, MouseDrag, TypeText


class TestActionConfig:
    """Test ActionConfig functionality."""

    def test_fluent_interface(self):
        """Test fluent configuration interface using Builder pattern."""
        config = ClickOptionsBuilder().set_pause_before_begin(0.5).set_pause_after_end(1.0).build()

        assert config.get_pause_before_begin() == 0.5
        assert config.get_pause_after_end() == 1.0


class TestClickOptions:
    """Test ClickOptions functionality."""

    def test_click_options_inherits_config(self):
        """Test that ClickOptions inherits from ActionConfig."""
        options = ClickOptionsBuilder().set_pause_before_begin(0.5).set_pause_after_end(1.0).build()

        # Should have ActionConfig methods
        assert options.get_pause_before_begin() == 0.5
        assert options.get_pause_after_end() == 1.0
        assert isinstance(options, ActionConfig)

    def test_double_click_configuration(self):
        """Test double-click configuration."""
        options = ClickOptionsBuilder().set_number_of_clicks(2).build()

        assert options.get_number_of_clicks() == 2


class TestPrimitives:
    """Test primitive actions."""

    def test_mouse_click_primitive(self):
        """Test MouseClick primitive."""
        from unittest.mock import Mock

        from qontinui.actions import ActionResultBuilder

        # Create a mock PureActions instance
        mock_pure = Mock()
        mock_pure.mouse_click.return_value = ActionResultBuilder().with_success(True).build()

        options = ClickOptionsBuilder().build()
        click = MouseClick(options, pure_actions=mock_pure)
        result = click.execute_at(100, 200)

        assert result.success is True
        mock_pure.mouse_click.assert_called()

    def test_mouse_drag_primitive(self):
        """Test MouseDrag primitive."""
        from unittest.mock import Mock

        from qontinui.actions import ActionResultBuilder

        # Create a mock PureActions instance
        mock_pure = Mock()
        mock_pure.mouse_down.return_value = ActionResultBuilder().with_success(True).build()
        mock_pure.mouse_move.return_value = ActionResultBuilder().with_success(True).build()
        mock_pure.mouse_up.return_value = ActionResultBuilder().with_success(True).build()

        options = DragOptionsBuilder().set_delay_after_drag(0.5).build()
        drag = MouseDrag(options, pure_actions=mock_pure)
        result = drag.execute_from_to(100, 200, 300, 400)

        assert result.success is True

    def test_type_text_primitive(self):
        """Test TypeText primitive."""
        from unittest.mock import Mock

        from qontinui.actions import ActionResultBuilder

        # Create a mock PureActions instance
        mock_pure = Mock()
        mock_pure.type_character.return_value = ActionResultBuilder().with_success(True).build()

        type_action = TypeText(pure_actions=mock_pure)
        result = type_action.execute_text("Hello")

        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
