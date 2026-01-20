"""Tests for UI-TARS executor."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from qontinui.extraction.runtime.uitars.config import UITARSSettings  # noqa: E402
from qontinui.extraction.runtime.uitars.executor import (  # noqa: E402
    HybridGrounder,
    LocalGroundingResult,
    UITARSExecutor,
)
from qontinui.extraction.runtime.uitars.models import (  # noqa: E402
    UITARSAction,
    UITARSActionType,
    UITARSInferenceResult,
    UITARSThought,
)


class MockLocalGrounder:
    """Mock local grounder for testing."""

    def __init__(self, x: int = 100, y: int = 200, confidence: float = 0.9):
        self.x = x
        self.y = y
        self.confidence = confidence
        self.found = True
        self.call_count = 0

    async def find(self, element_name: str, screenshot: np.ndarray) -> LocalGroundingResult:
        self.call_count += 1
        return LocalGroundingResult(
            x=self.x,
            y=self.y,
            confidence=self.confidence,
            found=self.found,
        )


class TestUITARSExecutor:
    """Test UITARSExecutor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.settings = UITARSSettings(
            execution_mode="uitars",
            confidence_threshold=0.7,
        )
        self.mock_provider = MagicMock()
        self.mock_provider.is_available.return_value = True
        self.executor = UITARSExecutor(self.mock_provider, self.settings)

    def test_create_executor(self):
        """Test creating executor instance."""
        assert self.executor.provider == self.mock_provider
        assert self.executor.settings == self.settings
        assert self.executor._local_grounder is None

    def test_from_settings_creates_executor(self):
        """Test factory method creates executor."""
        with patch("qontinui.extraction.runtime.uitars.executor.create_provider") as mock_create:
            mock_create.return_value = MagicMock()
            executor = UITARSExecutor.from_settings()

            mock_create.assert_called_once()
            assert isinstance(executor, UITARSExecutor)

    def test_set_local_grounder(self):
        """Test setting local grounder."""
        grounder = MockLocalGrounder()
        self.executor.set_local_grounder(grounder)

        assert self.executor._local_grounder == grounder

    @pytest.mark.asyncio
    async def test_ground_element_success(self):
        """Test successful element grounding."""
        # Set up mock inference result
        thought = UITARSThought(reasoning="Found the submit button")
        action = UITARSAction(
            action_type=UITARSActionType.CLICK,
            x=500,
            y=300,
            confidence=0.95,
        )
        self.mock_provider.infer.return_value = UITARSInferenceResult(
            thought=thought,
            action=action,
            raw_output="Test output",
            inference_time_ms=100.0,
        )

        screenshot = np.zeros((480, 640, 3), dtype=np.uint8)
        result = await self.executor.ground_element(screenshot, "Submit button")

        assert result.x == 500
        assert result.y == 300
        assert result.confidence == 0.95
        assert result.element_description == "Submit button"

    @pytest.mark.asyncio
    async def test_ground_element_failure(self):
        """Test failed element grounding."""
        # Set up mock with no coordinates
        thought = UITARSThought(reasoning="Cannot find element")
        action = UITARSAction(action_type=UITARSActionType.WAIT)
        self.mock_provider.infer.return_value = UITARSInferenceResult(
            thought=thought,
            action=action,
            raw_output="Test output",
            inference_time_ms=100.0,
        )

        screenshot = np.zeros((480, 640, 3), dtype=np.uint8)
        result = await self.executor.ground_element(screenshot, "Missing element")

        assert result.x == 0
        assert result.y == 0
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_execute_action_success(self):
        """Test successful action execution."""
        import sys

        thought = UITARSThought(reasoning="Clicking submit")
        action = UITARSAction(
            action_type=UITARSActionType.CLICK,
            x=100,
            y=200,
        )
        self.mock_provider.infer.return_value = UITARSInferenceResult(
            thought=thought,
            action=action,
            raw_output="Test",
            inference_time_ms=100.0,
        )

        screenshot = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_pyautogui = MagicMock()

        with patch.dict(sys.modules, {"pyautogui": mock_pyautogui}):
            result = await self.executor.execute_action(screenshot, "Click the submit button")

            assert result.success is True
            assert result.thought == thought
            assert result.action == action
            mock_pyautogui.click.assert_called_once_with(100, 200)

    @pytest.mark.asyncio
    async def test_hybrid_ground_local_success(self):
        """Test hybrid grounding when local succeeds."""
        grounder = MockLocalGrounder(x=150, y=250, confidence=0.85)
        self.executor.set_local_grounder(grounder)

        screenshot = np.zeros((100, 100, 3), dtype=np.uint8)
        result = await self.executor.hybrid_ground("Test element", screenshot)

        assert result.x == 150
        assert result.y == 250
        assert result.confidence == 0.85
        assert grounder.call_count == 1
        # Provider should not be called
        self.mock_provider.infer.assert_not_called()

    @pytest.mark.asyncio
    async def test_hybrid_ground_fallback_to_uitars(self):
        """Test hybrid grounding falls back to UI-TARS on low confidence."""
        # Local grounder with low confidence
        grounder = MockLocalGrounder(x=100, y=100, confidence=0.5)
        self.executor.set_local_grounder(grounder)

        # UI-TARS returns better result
        thought = UITARSThought(reasoning="Found element")
        action = UITARSAction(
            action_type=UITARSActionType.CLICK,
            x=200,
            y=200,
            confidence=0.9,
        )
        self.mock_provider.infer.return_value = UITARSInferenceResult(
            thought=thought,
            action=action,
            raw_output="Test",
            inference_time_ms=100.0,
        )

        screenshot = np.zeros((100, 100, 3), dtype=np.uint8)
        result = await self.executor.hybrid_ground(
            "Test element", screenshot, confidence_threshold=0.7
        )

        # Should use UI-TARS result
        assert result.x == 200
        assert result.y == 200
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_hybrid_ground_no_grounder(self):
        """Test hybrid grounding with no local grounder configured."""
        # Set up UI-TARS response
        thought = UITARSThought(reasoning="Found it")
        action = UITARSAction(
            action_type=UITARSActionType.CLICK,
            x=300,
            y=400,
            confidence=0.8,
        )
        self.mock_provider.infer.return_value = UITARSInferenceResult(
            thought=thought,
            action=action,
            raw_output="Test",
            inference_time_ms=100.0,
        )

        screenshot = np.zeros((100, 100, 3), dtype=np.uint8)
        result = await self.executor.hybrid_ground("Element", screenshot)

        assert result.x == 300
        assert result.y == 400


class TestHybridGrounder:
    """Test HybridGrounder class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.settings = UITARSSettings(
            execution_mode="hybrid",
            confidence_threshold=0.7,
            uitars_fallback_enabled=True,
        )

    @pytest.mark.asyncio
    async def test_local_mode_uses_local_only(self):
        """Test local mode only uses local grounder."""
        settings = UITARSSettings(execution_mode="local")
        local_grounder = MockLocalGrounder(x=100, y=200, confidence=0.9)

        grounder = HybridGrounder(
            local_grounder=local_grounder,
            uitars_executor=None,
            settings=settings,
        )

        screenshot = np.zeros((100, 100, 3), dtype=np.uint8)
        result = await grounder.ground("Test", screenshot)

        assert result.x == 100
        assert result.y == 200
        assert local_grounder.call_count == 1

    @pytest.mark.asyncio
    async def test_local_mode_no_grounder_returns_zero(self):
        """Test local mode without grounder returns zero confidence."""
        settings = UITARSSettings(execution_mode="local")

        grounder = HybridGrounder(
            local_grounder=None,
            uitars_executor=None,
            settings=settings,
        )

        screenshot = np.zeros((100, 100, 3), dtype=np.uint8)
        result = await grounder.ground("Test", screenshot)

        assert result.x == 0
        assert result.y == 0
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_uitars_mode_uses_uitars_only(self):
        """Test uitars mode only uses UI-TARS executor."""
        settings = UITARSSettings(execution_mode="uitars")

        mock_executor = MagicMock()
        mock_executor.ground_element = AsyncMock()
        mock_executor.ground_element.return_value = MagicMock(x=300, y=400, confidence=0.85)

        grounder = HybridGrounder(
            local_grounder=MockLocalGrounder(),
            uitars_executor=mock_executor,
            settings=settings,
        )

        screenshot = np.zeros((100, 100, 3), dtype=np.uint8)
        result = await grounder.ground("Test", screenshot)

        mock_executor.ground_element.assert_called_once()
        assert result.x == 300
        assert result.y == 400

    @pytest.mark.asyncio
    async def test_uitars_mode_no_executor_returns_zero(self):
        """Test uitars mode without executor returns zero confidence."""
        settings = UITARSSettings(execution_mode="uitars")

        grounder = HybridGrounder(
            local_grounder=None,
            uitars_executor=None,
            settings=settings,
        )

        screenshot = np.zeros((100, 100, 3), dtype=np.uint8)
        result = await grounder.ground("Test", screenshot)

        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_hybrid_mode_local_high_confidence(self):
        """Test hybrid mode uses local when confidence is high."""
        local_grounder = MockLocalGrounder(x=100, y=200, confidence=0.9)

        mock_executor = MagicMock()
        mock_executor.ground_element = AsyncMock()

        grounder = HybridGrounder(
            local_grounder=local_grounder,
            uitars_executor=mock_executor,
            settings=self.settings,
        )

        screenshot = np.zeros((100, 100, 3), dtype=np.uint8)
        result = await grounder.ground("Test", screenshot)

        assert result.x == 100
        assert result.y == 200
        # UI-TARS should not be called
        mock_executor.ground_element.assert_not_called()

    @pytest.mark.asyncio
    async def test_hybrid_mode_fallback_on_low_confidence(self):
        """Test hybrid mode falls back to UI-TARS on low confidence."""
        local_grounder = MockLocalGrounder(x=100, y=200, confidence=0.5)

        mock_executor = MagicMock()
        mock_executor.ground_element = AsyncMock()
        mock_executor.ground_element.return_value = MagicMock(x=300, y=400, confidence=0.9)

        grounder = HybridGrounder(
            local_grounder=local_grounder,
            uitars_executor=mock_executor,
            settings=self.settings,
        )

        screenshot = np.zeros((100, 100, 3), dtype=np.uint8)
        result = await grounder.ground("Test", screenshot)

        # Should use UI-TARS result
        assert result.x == 300
        assert result.y == 400
        mock_executor.ground_element.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_mode_fallback_disabled(self):
        """Test hybrid mode respects fallback disabled setting."""
        settings = UITARSSettings(
            execution_mode="hybrid",
            confidence_threshold=0.7,
            uitars_fallback_enabled=False,
        )
        local_grounder = MockLocalGrounder(x=100, y=200, confidence=0.5)

        mock_executor = MagicMock()
        mock_executor.ground_element = AsyncMock()

        grounder = HybridGrounder(
            local_grounder=local_grounder,
            uitars_executor=mock_executor,
            settings=settings,
        )

        screenshot = np.zeros((100, 100, 3), dtype=np.uint8)
        result = await grounder.ground("Test", screenshot)

        # Should return local result even with low confidence
        assert result.x == 100
        assert result.y == 200
        # UI-TARS should NOT be called
        mock_executor.ground_element.assert_not_called()


class TestExecutePyautoguiAction:
    """Test pyautogui action execution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.settings = UITARSSettings()
        self.mock_provider = MagicMock()
        self.mock_provider.is_available.return_value = True
        self.executor = UITARSExecutor(self.mock_provider, self.settings)

    @pytest.mark.asyncio
    async def test_execute_click_action(self):
        """Test executing click action."""
        action = UITARSAction(
            action_type=UITARSActionType.CLICK,
            x=100,
            y=200,
        )

        import sys

        mock_pyautogui = MagicMock()
        with patch.dict(sys.modules, {"pyautogui": mock_pyautogui}):
            result = await self.executor._execute_pyautogui_action(action)

            assert result is True
            mock_pyautogui.click.assert_called_once_with(100, 200)

    @pytest.mark.asyncio
    async def test_execute_double_click_action(self):
        """Test executing double click action."""
        action = UITARSAction(
            action_type=UITARSActionType.DOUBLE_CLICK,
            x=150,
            y=250,
        )

        import sys

        mock_pyautogui = MagicMock()
        with patch.dict(sys.modules, {"pyautogui": mock_pyautogui}):
            result = await self.executor._execute_pyautogui_action(action)

            assert result is True
            mock_pyautogui.doubleClick.assert_called_once_with(150, 250)

    @pytest.mark.asyncio
    async def test_execute_type_action(self):
        """Test executing type action."""
        action = UITARSAction(
            action_type=UITARSActionType.TYPE,
            x=100,
            y=100,
            text="hello world",
        )

        import sys

        mock_pyautogui = MagicMock()
        with patch.dict(sys.modules, {"pyautogui": mock_pyautogui}):
            result = await self.executor._execute_pyautogui_action(action)

            assert result is True
            mock_pyautogui.click.assert_called_once_with(100, 100)
            mock_pyautogui.typewrite.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_scroll_down_action(self):
        """Test executing scroll down action."""
        action = UITARSAction(
            action_type=UITARSActionType.SCROLL,
            scroll_direction="down",
            scroll_amount=100,
        )

        import sys

        mock_pyautogui = MagicMock()
        with patch.dict(sys.modules, {"pyautogui": mock_pyautogui}):
            result = await self.executor._execute_pyautogui_action(action)

            assert result is True
            mock_pyautogui.scroll.assert_called_once_with(-100)

    @pytest.mark.asyncio
    async def test_execute_scroll_up_action(self):
        """Test executing scroll up action."""
        action = UITARSAction(
            action_type=UITARSActionType.SCROLL,
            scroll_direction="up",
            scroll_amount=50,
        )

        import sys

        mock_pyautogui = MagicMock()
        with patch.dict(sys.modules, {"pyautogui": mock_pyautogui}):
            result = await self.executor._execute_pyautogui_action(action)

            assert result is True
            mock_pyautogui.scroll.assert_called_once_with(50)

    @pytest.mark.asyncio
    async def test_execute_hotkey_action(self):
        """Test executing hotkey action."""
        action = UITARSAction(
            action_type=UITARSActionType.HOTKEY,
            keys=["ctrl", "c"],
        )

        import sys

        mock_pyautogui = MagicMock()
        with patch.dict(sys.modules, {"pyautogui": mock_pyautogui}):
            result = await self.executor._execute_pyautogui_action(action)

            assert result is True
            mock_pyautogui.hotkey.assert_called_once_with("ctrl", "c")

    @pytest.mark.asyncio
    async def test_execute_hover_action(self):
        """Test executing hover action."""
        action = UITARSAction(
            action_type=UITARSActionType.HOVER,
            x=300,
            y=400,
        )

        import sys

        mock_pyautogui = MagicMock()
        with patch.dict(sys.modules, {"pyautogui": mock_pyautogui}):
            result = await self.executor._execute_pyautogui_action(action)

            assert result is True
            mock_pyautogui.moveTo.assert_called_once_with(300, 400)

    @pytest.mark.asyncio
    async def test_execute_drag_action(self):
        """Test executing drag action."""
        action = UITARSAction(
            action_type=UITARSActionType.DRAG,
            x=100,
            y=100,
            end_x=200,
            end_y=150,
        )

        import sys

        mock_pyautogui = MagicMock()
        with patch.dict(sys.modules, {"pyautogui": mock_pyautogui}):
            result = await self.executor._execute_pyautogui_action(action)

            assert result is True
            mock_pyautogui.moveTo.assert_called_once_with(100, 100)
            mock_pyautogui.drag.assert_called_once_with(100, 50)

    @pytest.mark.asyncio
    async def test_execute_wait_action(self):
        """Test executing wait action."""
        action = UITARSAction(
            action_type=UITARSActionType.WAIT,
            duration=0.01,  # Short duration for test
        )

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.return_value = None
            result = await self.executor._execute_pyautogui_action(action)

            assert result is True

    @pytest.mark.asyncio
    async def test_execute_done_action(self):
        """Test executing done action (no-op)."""
        action = UITARSAction(action_type=UITARSActionType.DONE)

        import sys

        mock_pyautogui = MagicMock()
        with patch.dict(sys.modules, {"pyautogui": mock_pyautogui}):
            result = await self.executor._execute_pyautogui_action(action)

            assert result is True
            # No pyautogui methods should be called
            mock_pyautogui.click.assert_not_called()
