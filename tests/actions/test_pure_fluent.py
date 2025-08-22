"""Tests for pure actions and fluent API following Brobot principles."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock pyautogui before importing
if 'pyautogui' not in sys.modules:
    sys.modules['pyautogui'] = MagicMock()

from qontinui.actions import PureActions, FluentActions, ActionResult


class TestPureActions:
    """Test pure atomic actions."""
    
    @pytest.fixture
    def actions(self):
        """Create PureActions instance."""
        with patch('qontinui.actions.pure.pyautogui') as mock_pyautogui:
            return PureActions()
    
    @patch('qontinui.actions.pure.pyautogui.mouseDown')
    def test_mouse_down(self, mock_mouse_down):
        """Test atomic mouse down action."""
        actions = PureActions()
        result = actions.mouse_down(100, 200, 'left')
        
        assert result.success == True
        mock_mouse_down.assert_called_once_with(100, 200, button='left')
    
    @patch('qontinui.actions.pure.pyautogui.mouseUp')
    def test_mouse_up(self, mock_mouse_up):
        """Test atomic mouse up action."""
        actions = PureActions()
        result = actions.mouse_up(100, 200, 'left')
        
        assert result.success == True
        mock_mouse_up.assert_called_once_with(100, 200, button='left')
    
    @patch('qontinui.actions.pure.pyautogui.moveTo')
    def test_mouse_move(self, mock_move_to):
        """Test atomic mouse move action."""
        actions = PureActions()
        result = actions.mouse_move(300, 400, duration=0.5)
        
        assert result.success == True
        assert result.data == (300, 400)
        mock_move_to.assert_called_once_with(300, 400, duration=0.5)
    
    @patch('qontinui.actions.pure.pyautogui.click')
    def test_mouse_click(self, mock_click):
        """Test atomic click action."""
        actions = PureActions()
        result = actions.mouse_click(150, 250, 'right')
        
        assert result.success == True
        assert result.data == (150, 250)
        mock_click.assert_called_once_with(150, 250, button='right')
    
    @patch('qontinui.actions.pure.pyautogui.keyDown')
    def test_key_down(self, mock_key_down):
        """Test atomic key down action."""
        actions = PureActions()
        result = actions.key_down('ctrl')
        
        assert result.success == True
        assert result.data == 'ctrl'
        mock_key_down.assert_called_once_with('ctrl')
    
    @patch('qontinui.actions.pure.pyautogui.keyUp')
    def test_key_up(self, mock_key_up):
        """Test atomic key up action."""
        actions = PureActions()
        result = actions.key_up('ctrl')
        
        assert result.success == True
        assert result.data == 'ctrl'
        mock_key_up.assert_called_once_with('ctrl')
    
    @patch('qontinui.actions.pure.pyautogui.typewrite')
    def test_type_character(self, mock_typewrite):
        """Test atomic character typing."""
        actions = PureActions()
        result = actions.type_character('a')
        
        assert result.success == True
        assert result.data == 'a'
        mock_typewrite.assert_called_once_with('a')
        
        # Test multi-character rejection
        result = actions.type_character('abc')
        assert result.success == False
        assert "Must be single character" in result.error


class TestFluentActions:
    """Test fluent API for action chaining."""
    
    @pytest.fixture
    def actions(self):
        """Create FluentActions instance."""
        with patch('qontinui.actions.pure.pyautogui'):
            return FluentActions()
    
    def test_fluent_chain_building(self, actions):
        """Test building a fluent action chain."""
        # Build chain
        actions.at(100, 200).click().wait(0.5).type_text("Hello")
        
        # Check chain has actions
        assert len(actions.chain.actions) > 0
    
    @patch('qontinui.actions.pure.pyautogui')
    def test_drag_composite_action(self, mock_pyautogui):
        """Test that drag is properly composed from atomic actions."""
        actions = FluentActions()
        
        # Build drag operation
        actions.at(100, 200).drag_to(300, 400, duration=1.0)
        
        # Drag should be composed of: mouseDown + moveTo + mouseUp
        assert len(actions.chain.actions) == 3
        
        # Execute the chain
        actions.execute()
        
        # Verify atomic actions were called
        mock_pyautogui.mouseDown.assert_called()
        mock_pyautogui.moveTo.assert_called()
        mock_pyautogui.mouseUp.assert_called()
    
    @patch('qontinui.actions.pure.pyautogui')
    def test_double_click_composite(self, mock_pyautogui):
        """Test that double click is composed from atomic clicks."""
        actions = FluentActions()
        
        # Build double click
        actions.at(150, 250).double_click()
        
        # Should be: click + wait + click
        assert len(actions.chain.actions) == 3
        
        actions.execute()
        
        # Verify two clicks
        assert mock_pyautogui.click.call_count == 2
    
    @patch('qontinui.actions.pure.pyautogui')
    def test_hotkey_composite(self, mock_pyautogui):
        """Test that hotkey is composed from key down/up actions."""
        actions = FluentActions()
        
        # Build hotkey
        actions.hotkey('ctrl', 'shift', 's')
        
        # Should be: 3 keyDowns + 3 keyUps (in reverse order)
        assert len(actions.chain.actions) == 6
        
        actions.execute()
        
        # Verify key operations
        assert mock_pyautogui.keyDown.call_count == 3
        assert mock_pyautogui.keyUp.call_count == 3
    
    @patch('qontinui.actions.pure.pyautogui')
    def test_complex_chain(self, mock_pyautogui):
        """Test a complex action chain."""
        actions = FluentActions()
        
        # Build complex chain
        results = (actions
            .at(100, 200)
            .click()
            .wait(0.5)
            .move_to(200, 300)
            .type_text("Test")
            .key('enter')
            .execute())
        
        # Verify execution
        assert len(results) > 0
        mock_pyautogui.click.assert_called()
        mock_pyautogui.moveTo.assert_called()
        mock_pyautogui.typewrite.assert_called()
        mock_pyautogui.press.assert_called()
    
    def test_clear_chain(self, actions):
        """Test clearing the action chain."""
        actions.at(100, 200).click().wait(1.0)
        assert len(actions.chain.actions) > 0
        
        actions.clear()
        assert len(actions.chain.actions) == 0
        assert actions._current_position is None
    
    @patch('qontinui.actions.pure.pyautogui')
    def test_action_result_tracking(self, mock_pyautogui):
        """Test that action results are properly tracked."""
        actions = FluentActions()
        
        # Make all actions succeed
        mock_pyautogui.click.return_value = None
        mock_pyautogui.typewrite.return_value = None
        
        actions.at(100, 200).click().type_text("Hi").execute()
        
        # Check success
        assert actions.success == True
        assert len(actions.results) > 0
        
        for result in actions.results:
            assert isinstance(result, ActionResult)
            assert result.success == True