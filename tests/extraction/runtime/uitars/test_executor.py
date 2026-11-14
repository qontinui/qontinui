"""Tests for UI-TARS executor."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from tests._env_probe import HAS_FUNCTIONAL_DISPLAY

# The executor takes a screenshot via mss before calling the (mocked)
# inference / HAL path; on headless Linux the mss screenshot
# fails at xcb.connect(display=None) regardless of mocks.
_NEEDS_DISPLAY = pytest.mark.skipif(
    not HAS_FUNCTIONAL_DISPLAY,
    reason="executor's screenshot path uses mss; needs a functional display",
)

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
from qontinui.hal.interfaces.input_controller import (  # noqa: E402
    MouseButton,
    MousePosition,
)
from qontinui.hal.interfaces.keyboard_controller import (  # noqa: E402
    IKeyboardController,
)
from qontinui.hal.interfaces.mouse_controller import IMouseController  # noqa: E402


class MockLocalGrounder:
    """Mock local grounder for testing."""

    def __init__(self, x: int = 100, y: int = 200, confidence: float = 0.9):
        self.x = x
        self.y = y
        self.confidence = confidence
        self.found = True
        self.call_count = 0

    async def find(
        self, element_name: str, screenshot: np.ndarray
    ) -> LocalGroundingResult:
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
        with patch(
            "qontinui.extraction.runtime.uitars.executor.create_provider"
        ) as mock_create:
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
    @_NEEDS_DISPLAY
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
    @_NEEDS_DISPLAY
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
    @_NEEDS_DISPLAY
    async def test_execute_action_success(self):
        """Test successful action execution."""
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
        mouse, keyboard = _make_hal_mocks()
        self.executor._hal_mouse = mouse
        self.executor._hal_keyboard = keyboard

        result = await self.executor.execute_action(screenshot, "Click the submit button")

        assert result.success is True
        assert result.thought == thought
        assert result.action == action
        mouse.click_at.assert_called_once_with(100, 200)

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
    @_NEEDS_DISPLAY
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
    @_NEEDS_DISPLAY
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
        mock_executor.ground_element.return_value = MagicMock(
            x=300, y=400, confidence=0.85
        )

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
        mock_executor.ground_element.return_value = MagicMock(
            x=300, y=400, confidence=0.9
        )

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


def _make_hal_mocks(position: tuple[int, int] = (500, 500)) -> tuple[MagicMock, MagicMock]:
    """Mock HAL mouse/keyboard controllers with the pointer at ``position``."""
    mouse = MagicMock(spec=IMouseController)
    mouse.get_mouse_position.return_value = MousePosition(*position)
    keyboard = MagicMock(spec=IKeyboardController)
    return mouse, keyboard


class TestExecuteHalAction:
    """Test HAL action execution (UI-TARS action -> HAL controller call)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.settings = UITARSSettings()
        self.mock_provider = MagicMock()
        self.mock_provider.is_available.return_value = True
        self.mouse, self.keyboard = _make_hal_mocks()
        self.executor = UITARSExecutor(
            self.mock_provider,
            self.settings,
            hal_mouse=self.mouse,
            hal_keyboard=self.keyboard,
        )

    @pytest.mark.asyncio
    async def test_execute_click_action(self):
        """Click maps to click_at."""
        action = UITARSAction(action_type=UITARSActionType.CLICK, x=100, y=200)

        result = await self.executor._execute_hal_action(action)

        assert result is True
        self.mouse.click_at.assert_called_once_with(100, 200)

    @pytest.mark.asyncio
    async def test_execute_double_click_action(self):
        """Double click maps to double_click_at."""
        action = UITARSAction(
            action_type=UITARSActionType.DOUBLE_CLICK, x=150, y=250
        )

        result = await self.executor._execute_hal_action(action)

        assert result is True
        self.mouse.double_click_at.assert_called_once_with(150, 250)

    @pytest.mark.asyncio
    async def test_execute_right_click_action(self):
        """Right click maps to click_at with the right button."""
        action = UITARSAction(action_type=UITARSActionType.RIGHT_CLICK, x=10, y=20)

        result = await self.executor._execute_hal_action(action)

        assert result is True
        self.mouse.click_at.assert_called_once_with(10, 20, button=MouseButton.RIGHT)

    @pytest.mark.asyncio
    async def test_execute_type_action(self):
        """Type clicks the target, then types the text."""
        action = UITARSAction(
            action_type=UITARSActionType.TYPE, x=100, y=100, text="hello world"
        )

        result = await self.executor._execute_hal_action(action)

        assert result is True
        self.mouse.click_at.assert_called_once_with(100, 100)
        self.keyboard.type_text.assert_called_once_with("hello world", interval=0.02)

    @pytest.mark.asyncio
    async def test_execute_scroll_down_action(self):
        """Scroll down is a negative vertical scroll."""
        action = UITARSAction(
            action_type=UITARSActionType.SCROLL,
            scroll_direction="down",
            scroll_amount=100,
        )

        result = await self.executor._execute_hal_action(action)

        assert result is True
        self.mouse.mouse_scroll.assert_called_once_with(-100, horizontal_clicks=0)

    @pytest.mark.asyncio
    async def test_execute_scroll_up_action(self):
        """Scroll up is a positive vertical scroll."""
        action = UITARSAction(
            action_type=UITARSActionType.SCROLL,
            scroll_direction="up",
            scroll_amount=50,
        )

        result = await self.executor._execute_hal_action(action)

        assert result is True
        self.mouse.mouse_scroll.assert_called_once_with(50, horizontal_clicks=0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("direction", "expected"), [("left", -30), ("right", 30)])
    async def test_execute_horizontal_scroll_action(self, direction, expected):
        """Left/right scroll uses the horizontal axis only."""
        action = UITARSAction(
            action_type=UITARSActionType.SCROLL,
            scroll_direction=direction,
            scroll_amount=30,
        )

        result = await self.executor._execute_hal_action(action)

        assert result is True
        self.mouse.mouse_scroll.assert_called_once_with(0, horizontal_clicks=expected)

    @pytest.mark.asyncio
    async def test_execute_hotkey_action(self):
        """Hotkey maps to keyboard.hotkey."""
        action = UITARSAction(action_type=UITARSActionType.HOTKEY, keys=["ctrl", "c"])

        result = await self.executor._execute_hal_action(action)

        assert result is True
        self.keyboard.hotkey.assert_called_once_with("ctrl", "c")

    @pytest.mark.asyncio
    async def test_execute_hover_action(self):
        """Hover maps to move_mouse."""
        action = UITARSAction(action_type=UITARSActionType.HOVER, x=300, y=400)

        result = await self.executor._execute_hal_action(action)

        assert result is True
        self.mouse.move_mouse.assert_called_once_with(300, 400)

    @pytest.mark.asyncio
    async def test_execute_drag_action(self):
        """Drag moves to the start, then drags start -> end."""
        action = UITARSAction(
            action_type=UITARSActionType.DRAG, x=100, y=100, end_x=200, end_y=150
        )

        result = await self.executor._execute_hal_action(action)

        assert result is True
        self.mouse.move_mouse.assert_called_once_with(100, 100)
        self.mouse.drag.assert_called_once_with(100, 100, 200, 150)

    @pytest.mark.asyncio
    async def test_execute_wait_action(self):
        """Wait sleeps and touches no controller."""
        action = UITARSAction(action_type=UITARSActionType.WAIT, duration=0.01)

        with patch(
            "qontinui.extraction.runtime.uitars.hal_dispatch.asyncio.sleep",
            new=AsyncMock(),
        ) as mock_sleep:
            result = await self.executor._execute_hal_action(action)

        assert result is True
        mock_sleep.assert_awaited_once_with(0.01)
        assert self.mouse.mock_calls == []
        assert self.keyboard.mock_calls == []

    @pytest.mark.asyncio
    async def test_execute_done_action(self):
        """Done is a no-op."""
        action = UITARSAction(action_type=UITARSActionType.DONE)

        result = await self.executor._execute_hal_action(action)

        assert result is True
        assert self.mouse.mock_calls == []
        assert self.keyboard.mock_calls == []

    @pytest.mark.asyncio
    async def test_corner_abort_refuses_action(self, caplog):
        """Pointer in the (0, 0) failsafe corner refuses the action with an error."""
        self.mouse.get_mouse_position.return_value = MousePosition(0, 0)
        action = UITARSAction(action_type=UITARSActionType.CLICK, x=100, y=200)

        with caplog.at_level("ERROR"):
            result = await self.executor._execute_hal_action(action)

        assert result is False
        self.mouse.click_at.assert_not_called()
        assert "failsafe corner" in caplog.text

    @pytest.mark.asyncio
    async def test_corner_abort_between_compound_steps(self):
        """Pointer moved into the corner after step one stops step two."""
        self.mouse.get_mouse_position.side_effect = [
            MousePosition(500, 500),
            MousePosition(0, 0),
        ]
        action = UITARSAction(
            action_type=UITARSActionType.DRAG, x=100, y=100, end_x=200, end_y=150
        )

        with patch(
            "qontinui.extraction.runtime.uitars.hal_dispatch._pause_between_steps",
            new=AsyncMock(),
        ):
            result = await self.executor._execute_hal_action(action)

        assert result is False
        self.mouse.move_mouse.assert_called_once_with(100, 100)
        self.mouse.drag.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("action", "first_step", "second_step"),
        [
            (
                UITARSAction(
                    action_type=UITARSActionType.TYPE, x=1, y=2, text="abc"
                ),
                "mouse.click_at",
                "keyboard.type_text",
            ),
            (
                UITARSAction(
                    action_type=UITARSActionType.DRAG, x=1, y=2, end_x=3, end_y=4
                ),
                "mouse.move_mouse",
                "mouse.drag",
            ),
        ],
    )
    async def test_compound_action_pauses_between_steps(
        self, action, first_step, second_step
    ):
        """Compound actions sleep COMPOUND_STEP_PAUSE_S between their two steps."""
        from qontinui.extraction.runtime.uitars import hal_dispatch

        manager = MagicMock()
        manager.attach_mock(self.mouse, "mouse")
        manager.attach_mock(self.keyboard, "keyboard")
        sleep = AsyncMock()
        manager.attach_mock(sleep, "sleep")

        with patch.object(hal_dispatch.asyncio, "sleep", new=sleep):
            result = await self.executor._execute_hal_action(action)

        assert result is True
        assert hal_dispatch.COMPOUND_STEP_PAUSE_S == 0.1
        names = [c[0] for c in manager.mock_calls if c[0] != "mouse.get_mouse_position"]
        assert names == [first_step, "sleep", second_step]
        sleep.assert_awaited_once_with(0.1)

    @pytest.mark.asyncio
    async def test_defaults_to_hal_factory_controllers(self):
        """Without injected controllers the HAL factory's are used."""
        executor = UITARSExecutor(self.mock_provider, self.settings)
        action = UITARSAction(action_type=UITARSActionType.HOVER, x=7, y=8)

        with (
            patch(
                "qontinui.hal.factory.HALFactory.get_mouse_controller",
                return_value=self.mouse,
            ),
            patch(
                "qontinui.hal.factory.HALFactory.get_keyboard_controller",
                return_value=self.keyboard,
            ),
        ):
            result = await executor._execute_hal_action(action)

        assert result is True
        self.mouse.move_mouse.assert_called_once_with(7, 8)


class TestExplorerHalAction:
    """The explorer dispatches through the same HAL path."""

    @pytest.mark.asyncio
    async def test_explorer_dispatches_through_hal(self):
        from qontinui.extraction.runtime.uitars.explorer import UITARSExplorer

        mouse, keyboard = _make_hal_mocks()
        explorer = UITARSExplorer(
            settings=UITARSSettings(),
            provider=MagicMock(),
            hal_mouse=mouse,
            hal_keyboard=keyboard,
        )

        assert await explorer._execute_hal_action("click", x=5, y=6) is True
        assert (
            await explorer._execute_hal_action(
                "scroll", scroll_direction="right", scroll_amount=4
            )
            is True
        )
        assert await explorer._execute_hal_action("hotkey", keys=["ctrl", "v"]) is True

        mouse.click_at.assert_called_once_with(5, 6)
        mouse.mouse_scroll.assert_called_once_with(0, horizontal_clicks=4)
        keyboard.hotkey.assert_called_once_with("ctrl", "v")

    @pytest.mark.asyncio
    async def test_explorer_unknown_action_returns_false(self):
        from qontinui.extraction.runtime.uitars.explorer import UITARSExplorer

        mouse, keyboard = _make_hal_mocks()
        explorer = UITARSExplorer(provider=MagicMock(), hal_mouse=mouse, hal_keyboard=keyboard)

        assert await explorer._execute_hal_action("teleport", x=1, y=1) is False
        assert mouse.mock_calls == []
