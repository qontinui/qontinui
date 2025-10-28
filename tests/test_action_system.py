"""Test the Brobot-style action system."""

from unittest.mock import patch

import pytest

from qontinui.actions import (
    Action,
    ActionConfig,
    ActionResult,
    ClickOptions,
    DragOptions,
)
from qontinui.actions.basic.click.click_options import ClickOptionsBuilder
from qontinui.actions.composite.drag.drag_options import DragOptionsBuilder
from qontinui.primitives import MouseClick, MouseDrag, TypeText


class TestActionConfig:
    """Test ActionConfig functionality."""

    def test_fluent_interface(self):
        """Test fluent configuration interface using Builder pattern."""
        config = (
            ClickOptionsBuilder()
            .set_pause_before_begin(0.5)
            .set_pause_after_end(1.0)
            .build()
        )

        assert config.get_pause_before_begin() == 0.5
        assert config.get_pause_after_end() == 1.0


class TestClickOptions:
    """Test ClickOptions functionality."""

    def test_click_options_inherits_config(self):
        """Test that ClickOptions inherits from ActionConfig."""
        options = (
            ClickOptionsBuilder()
            .set_pause_before_begin(0.5)
            .set_pause_after_end(1.0)
            .build()
        )

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

    @patch("qontinui.actions.pure.PureActions.mouse_click")
    def test_mouse_click_primitive(self, mock_mouse_click):
        """Test MouseClick primitive."""
        # Mock PureActions.mouse_click to return success
        from qontinui.actions import ActionResultBuilder
        mock_mouse_click.return_value = ActionResultBuilder().with_success(True).build()

        options = ClickOptionsBuilder().build()
        click = MouseClick(options)
        result = click.execute_at(100, 200)

        assert result.success is True
        mock_mouse_click.assert_called()

    @patch("qontinui.actions.pure.PureActions.mouse_up")
    @patch("qontinui.actions.pure.PureActions.mouse_down")
    @patch("qontinui.actions.pure.PureActions.mouse_move")
    def test_mouse_drag_primitive(self, mock_mouse_move, mock_mouse_down, mock_mouse_up):
        """Test MouseDrag primitive."""
        # Mock PureActions methods to return success
        from qontinui.actions import ActionResultBuilder
        mock_mouse_down.return_value = ActionResultBuilder().with_success(True).build()
        mock_mouse_move.return_value = ActionResultBuilder().with_success(True).build()
        mock_mouse_up.return_value = ActionResultBuilder().with_success(True).build()

        options = DragOptionsBuilder().set_delay_after_drag(0.5).build()
        drag = MouseDrag(options)
        result = drag.execute_from_to(100, 200, 300, 400)

        assert result.success is True

    @patch("qontinui.actions.pure.PureActions.type_text")
    def test_type_text_primitive(self, mock_type_text):
        """Test TypeText primitive."""
        # Mock PureActions.type_text to return success
        from qontinui.actions import ActionResultBuilder
        mock_type_text.return_value = ActionResultBuilder().with_success(True).build()

        type_action = TypeText()
        result = type_action.execute_text("Hello")

        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
