"""Test the Brobot-style action system."""

import pytest
from unittest.mock import MagicMock, patch
from qontinui.actions import (
    Action, ActionConfig, ClickOptions, DragOptions,
    ActionResult, ActionLifecyclePhase
)
from qontinui.primitives import MouseClick, MouseDrag, TypeText


class TestActionConfig:
    """Test ActionConfig functionality."""
    
    def test_fluent_interface(self):
        """Test fluent configuration interface."""
        config = (ActionConfig()
            .pause_before(0.5)
            .pause_after(1.0)
            .max_attempts(3)
            .retry_on_failure(True))
        
        assert config._pause_before == 0.5
        assert config._pause_after == 1.0
        assert config._max_attempts == 3
        assert config._retry_on_failure is True
    
    def test_config_copy(self):
        """Test configuration copying."""
        config1 = ActionConfig().pause_before(1.0)
        config2 = config1.copy()
        config2.pause_after(2.0)
        
        assert config1._pause_before == 1.0
        assert config1._pause_after == 0.0  # Original unchanged
        assert config2._pause_before == 1.0
        assert config2._pause_after == 2.0
    
    def test_config_merge(self):
        """Test configuration merging."""
        base = ActionConfig().pause_before(1.0)
        override = ActionConfig().pause_after(2.0).max_attempts(5)
        
        base.merge(override)
        
        assert base._pause_before == 1.0  # Kept from base
        assert base._pause_after == 2.0   # Merged from override
        assert base._max_attempts == 5    # Merged from override


class TestClickOptions:
    """Test ClickOptions functionality."""
    
    def test_click_options_inherits_config(self):
        """Test that ClickOptions inherits from ActionConfig."""
        options = ClickOptions()
        
        # Should have ActionConfig methods
        options.pause_before(0.5).pause_after(1.0)
        
        assert options._pause_before == 0.5
        assert options._pause_after == 1.0
        assert options._action_name == "click"
    
    def test_double_click_configuration(self):
        """Test double-click configuration."""
        options = ClickOptions().double_click()
        
        assert options._click_count == 2
        assert options._click_type.value == "double"


class TestAction:
    """Test Action lifecycle management."""
    
    @patch('qontinui.actions.pure.pyautogui')
    def test_action_lifecycle_phases(self, mock_pyautogui):
        """Test that lifecycle phases are executed in order."""
        phases_executed = []
        
        action = Action(ActionConfig())
        
        # Add hooks for each phase
        action.add_lifecycle_hook(
            ActionLifecyclePhase.BEFORE_ACTION,
            lambda ctx: phases_executed.append("before")
        )
        action.add_lifecycle_hook(
            ActionLifecyclePhase.DURING_ACTION,
            lambda ctx: phases_executed.append("during")
        )
        action.add_lifecycle_hook(
            ActionLifecyclePhase.AFTER_ACTION,
            lambda ctx: phases_executed.append("after")
        )
        action.add_lifecycle_hook(
            ActionLifecyclePhase.ON_SUCCESS,
            lambda ctx: phases_executed.append("success")
        )
        
        # Execute a simple action
        result = action.click(100, 200)
        
        assert phases_executed == ["before", "during", "after", "success"]
    
    @patch('qontinui.actions.pure.pyautogui')
    def test_action_retry_on_failure(self, mock_pyautogui):
        """Test retry mechanism on failure."""
        attempt_count = 0
        
        def failing_action():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                return ActionResult(success=False, error="Test failure")
            return ActionResult(success=True)
        
        action = Action(ActionConfig().max_attempts(3).retry_on_failure(True))
        result = action.execute(failing_action)
        
        assert attempt_count == 3
        assert result.success is True
    
    @patch('qontinui.actions.pure.pyautogui')
    def test_action_pauses(self, mock_pyautogui):
        """Test that pauses are applied correctly."""
        import time
        
        config = ActionConfig().pause_before(0.1).pause_after(0.1)
        action = Action(config)
        
        start_time = time.time()
        action.click(100, 200)
        elapsed = time.time() - start_time
        
        # Should have at least pause_before + pause_after time
        assert elapsed >= 0.2


class TestPrimitives:
    """Test primitive actions."""
    
    @patch('qontinui.actions.pure.pyautogui')
    def test_mouse_click_primitive(self, mock_pyautogui):
        """Test MouseClick primitive."""
        mock_pyautogui.click.return_value = None
        
        click = MouseClick(ClickOptions())
        result = click.execute_at(100, 200)
        
        assert result.success is True
        mock_pyautogui.click.assert_called_with(100, 200, button='left')
    
    @patch('qontinui.actions.pure.pyautogui')
    def test_mouse_drag_primitive(self, mock_pyautogui):
        """Test MouseDrag primitive."""
        mock_pyautogui.moveTo.return_value = None
        mock_pyautogui.mouseDown.return_value = None
        mock_pyautogui.mouseUp.return_value = None
        
        drag = MouseDrag(DragOptions().drag_duration(0.5))
        result = drag.execute_from_to(100, 200, 300, 400)
        
        assert result.success is True
        assert result.data['start'] == (100, 200)
        assert result.data['end'] == (300, 400)
    
    @patch('qontinui.actions.pure.pyautogui')
    def test_type_text_primitive(self, mock_pyautogui):
        """Test TypeText primitive."""
        mock_pyautogui.typewrite.return_value = None
        
        type_action = TypeText()
        result = type_action.execute_text("Hello")
        
        assert result.success is True
        assert result.data['text'] == "Hello"
        assert result.data['length'] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])