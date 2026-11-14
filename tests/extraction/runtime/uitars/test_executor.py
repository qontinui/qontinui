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
from qontinui.extraction.runtime.uitars.hal_dispatch import (  # noqa: E402
    _resolve_failsafe_rect as _REAL_RESOLVE_FAILSAFE_RECT,
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
from qontinui.hal.interfaces.screen_capture import Monitor  # noqa: E402


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


FAILSAFE_RECT = (0, 0, 1920, 1080)


@pytest.fixture(autouse=True)
def _pin_dispatch_environment(monkeypatch):
    """Pin the platform (scroll units) and failsafe rectangle (corners).

    Without this, scroll assertions would depend on the OS running the tests
    and the failsafe on whatever display is attached.
    """
    from qontinui.extraction.runtime.uitars import hal_dispatch

    monkeypatch.setattr(hal_dispatch, "_current_platform", lambda: "linux")
    monkeypatch.setattr(hal_dispatch, "_resolve_failsafe_rect", lambda: FAILSAFE_RECT)


def _make_hal_mocks(position: tuple[int, int] = (500, 500)) -> tuple[MagicMock, MagicMock]:
    """Mock HAL mouse/keyboard controllers with the pointer at ``position``."""
    mouse = MagicMock(spec=IMouseController)
    mouse.get_mouse_position.return_value = MousePosition(*position)
    for name in ("click_at", "double_click_at", "move_mouse", "drag", "mouse_scroll"):
        getattr(mouse, name).return_value = True
    keyboard = MagicMock(spec=IKeyboardController)
    keyboard.type_text.return_value = True
    keyboard.hotkey.return_value = True
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
    @pytest.mark.parametrize("corner", [(0, 0), (0, 1079), (1919, 0), (1919, 1079)])
    async def test_corner_abort_refuses_action(self, caplog, corner):
        """Pointer on any of the four screen corners refuses the action."""
        self.mouse.get_mouse_position.return_value = MousePosition(*corner)
        action = UITARSAction(action_type=UITARSActionType.CLICK, x=100, y=200)

        with caplog.at_level("ERROR"):
            result = await self.executor._execute_hal_action(action)

        assert result is False
        self.mouse.click_at.assert_not_called()
        assert "failsafe corner" in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("position", [(1, 0), (0, 1), (1918, 1079), (960, 540)])
    async def test_near_corner_is_not_a_failsafe_point(self, position):
        """Only the exact corner pixels abort, as in pyautogui."""
        self.mouse.get_mouse_position.return_value = MousePosition(*position)
        action = UITARSAction(action_type=UITARSActionType.CLICK, x=100, y=200)

        assert await self.executor._execute_hal_action(action) is True
        self.mouse.click_at.assert_called_once_with(100, 200)

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
            "qontinui.extraction.runtime.uitars.hal_dispatch._pause_after_step",
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
    async def test_compound_action_pauses_after_each_step(
        self, action, first_step, second_step
    ):
        """Compound actions sleep STEP_PAUSE_S after each of their two steps."""
        from qontinui.extraction.runtime.uitars import hal_dispatch

        manager = MagicMock()
        manager.attach_mock(self.mouse, "mouse")
        manager.attach_mock(self.keyboard, "keyboard")
        sleep = AsyncMock()
        manager.attach_mock(sleep, "sleep")

        with patch.object(hal_dispatch.asyncio, "sleep", new=sleep):
            result = await self.executor._execute_hal_action(action)

        assert result is True
        assert hal_dispatch.STEP_PAUSE_S == 0.1
        names = [c[0] for c in manager.mock_calls if c[0] != "mouse.get_mouse_position"]
        assert names == [first_step, "sleep", second_step, "sleep"]
        assert [c.args for c in sleep.await_args_list] == [(0.1,), (0.1,)]

    @pytest.mark.asyncio
    async def test_execute_action_settles_before_screenshot_after(self, monkeypatch):
        """execute_action pauses STEP_PAUSE_S after the input before capturing."""
        from qontinui.extraction.runtime.uitars import hal_dispatch

        monkeypatch.delenv("QONTINUI_OMNIPARSER_PREFILTER", raising=False)
        self.mock_provider.infer.return_value = UITARSInferenceResult(
            thought=UITARSThought(reasoning="click"),
            action=UITARSAction(action_type=UITARSActionType.CLICK, x=10, y=20),
            raw_output="",
            inference_time_ms=1.0,
        )
        self.executor._initialized = True
        self.executor._screen_capture = MagicMock()
        capture = AsyncMock(return_value=np.zeros((2, 2, 3), dtype=np.uint8))
        sleep = AsyncMock()
        manager = MagicMock()
        manager.attach_mock(self.mouse, "mouse")
        manager.attach_mock(sleep, "sleep")
        manager.attach_mock(capture, "capture")

        with (
            patch.object(hal_dispatch.asyncio, "sleep", new=sleep),
            patch.object(self.executor, "_capture_screenshot", new=capture),
        ):
            result = await self.executor.execute_action(
                np.zeros((2, 2, 3), dtype=np.uint8), "click"
            )

        assert result.success is True
        names = [c[0] for c in manager.mock_calls if c[0] != "mouse.get_mouse_position"]
        assert names == ["mouse.click_at", "sleep", "capture"]
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


class TestScrollUnits:
    """UI-TARS scroll amounts keep their pyautogui meaning on every platform."""

    @pytest.mark.parametrize(
        ("platform", "amount", "expected"),
        [
            ("linux", 100, 100),
            ("linux", -3, -3),
            ("win32", 120, 1),
            ("win32", -240, -2),
            ("win32", 100, 1),
            ("win32", -100, -1),
            ("win32", 30, 1),
            ("win32", -1, -1),
            ("win32", 600, 5),
            ("win32", 0, 0),
        ],
    )
    def test_scroll_amount_to_clicks(self, platform, amount, expected):
        from qontinui.extraction.runtime.uitars.hal_dispatch import (
            scroll_amount_to_clicks,
        )

        assert scroll_amount_to_clicks(amount, platform=platform) == expected

    @pytest.mark.parametrize("amount", [5, -12])
    def test_darwin_passes_through_documenting_current_behaviour(self, amount):
        """Documents current behaviour, not correctness: on macOS pyautogui
        scrolled LINE units and pynput scrolls amount*10 PIXEL units, with no
        exact conversion, so the amount is passed through unconverted."""
        from qontinui.extraction.runtime.uitars.hal_dispatch import (
            scroll_amount_to_clicks,
        )

        assert scroll_amount_to_clicks(amount, platform="darwin") == amount

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("platform", "direction", "expected"),
        [
            ("win32", "down", (-2, 0)),
            ("win32", "right", (0, 2)),
            ("linux", "down", (-240, 0)),
            ("linux", "left", (0, -240)),
        ],
    )
    async def test_dispatch_converts_on_running_platform(
        self, monkeypatch, platform, direction, expected
    ):
        from qontinui.extraction.runtime.uitars import hal_dispatch

        monkeypatch.setattr(hal_dispatch, "_current_platform", lambda: platform)
        mouse, keyboard = _make_hal_mocks()

        result = await hal_dispatch.execute_hal_action(
            "scroll",
            scroll_direction=direction,
            scroll_amount=240,
            mouse=mouse,
            keyboard=keyboard,
        )

        assert result is True
        mouse.mouse_scroll.assert_called_once_with(
            expected[0], horizontal_clicks=expected[1]
        )


class TestControllerResults:
    """A controller call reporting failure fails the action."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("kwargs", "method"),
        [
            ({"action_type": "click", "x": 1, "y": 2}, "mouse.click_at"),
            ({"action_type": "double_click", "x": 1, "y": 2}, "mouse.double_click_at"),
            ({"action_type": "hover", "x": 1, "y": 2}, "mouse.move_mouse"),
            (
                {"action_type": "scroll", "scroll_direction": "up", "scroll_amount": 1},
                "mouse.mouse_scroll",
            ),
            ({"action_type": "hotkey", "keys": ["ctrl", "c"]}, "keyboard.hotkey"),
            ({"action_type": "type", "text": "abc"}, "keyboard.type_text"),
        ],
    )
    async def test_false_from_controller_is_false(self, kwargs, method):
        from qontinui.extraction.runtime.uitars.hal_dispatch import execute_hal_action

        mouse, keyboard = _make_hal_mocks()
        owner, name = method.split(".")
        getattr({"mouse": mouse, "keyboard": keyboard}[owner], name).return_value = False

        assert await execute_hal_action(**kwargs, mouse=mouse, keyboard=keyboard) is False

    @pytest.mark.asyncio
    async def test_failed_first_step_skips_second_step(self):
        from qontinui.extraction.runtime.uitars.hal_dispatch import execute_hal_action

        mouse, keyboard = _make_hal_mocks()
        mouse.click_at.return_value = False

        result = await execute_hal_action(
            "type", x=1, y=2, text="abc", mouse=mouse, keyboard=keyboard
        )

        assert result is False
        keyboard.type_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_move_skips_drag(self):
        from qontinui.extraction.runtime.uitars.hal_dispatch import execute_hal_action

        mouse, keyboard = _make_hal_mocks()
        mouse.move_mouse.return_value = False

        result = await execute_hal_action(
            "drag", x=1, y=2, end_x=3, end_y=4, mouse=mouse, keyboard=keyboard
        )

        assert result is False
        mouse.drag.assert_not_called()


class TestIncompleteActions:
    """Each caller keeps its historical answer for an action missing its inputs."""

    INCOMPLETE = [
        ("click", {}),
        ("double_click", {"x": 1}),
        ("right_click", {"y": 1}),
        ("hover", {}),
        ("drag", {"x": 1, "y": 2}),
        ("hotkey", {}),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("action_type", "kwargs"), INCOMPLETE)
    async def test_executor_reports_success(self, action_type, kwargs):
        mouse, keyboard = _make_hal_mocks()
        executor = UITARSExecutor(
            MagicMock(), UITARSSettings(), hal_mouse=mouse, hal_keyboard=keyboard
        )
        action = UITARSAction(action_type=UITARSActionType(action_type), **kwargs)

        assert await executor._execute_hal_action(action) is True
        assert [c[0] for c in mouse.mock_calls] == []
        assert keyboard.mock_calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("action_type", "kwargs"), INCOMPLETE)
    async def test_explorer_reports_failure(self, action_type, kwargs):
        from qontinui.extraction.runtime.uitars.explorer import UITARSExplorer

        mouse, keyboard = _make_hal_mocks()
        explorer = UITARSExplorer(
            provider=MagicMock(), hal_mouse=mouse, hal_keyboard=keyboard
        )

        assert await explorer._execute_hal_action(action_type, **kwargs) is False
        assert mouse.mock_calls == []
        assert keyboard.mock_calls == []

    @pytest.mark.asyncio
    async def test_unknown_scroll_direction_is_a_noop_success(self):
        from qontinui.extraction.runtime.uitars.hal_dispatch import execute_hal_action

        mouse, keyboard = _make_hal_mocks()

        result = await execute_hal_action(
            "scroll", scroll_direction="sideways", mouse=mouse, keyboard=keyboard
        )

        assert result is True
        mouse.mouse_scroll.assert_not_called()


def _monitor(index: int, x: int, y: int, width: int, height: int) -> Monitor:
    return Monitor(index=index, x=x, y=y, width=width, height=height)


# mss order on a two-monitor desk where the LEFT monitor is listed first and
# the primary (origin (0, 0)) second: mss monitors[1] / [2].
LEFT_THEN_PRIMARY = [
    _monitor(0, -1920, 0, 1920, 1080),
    _monitor(1, 0, 0, 2560, 1440),
]
# X11: monitors extend right of the origin, the X screen spans both.
X11_TWO_MONITORS = [
    _monitor(0, 1920, 0, 2560, 1440),
    _monitor(1, 0, 0, 1920, 1080),
]


class TestFailsafeRect:
    """Which rectangle's corners abort, per platform (pyautogui's size())."""

    def test_corners_of_a_rect(self):
        from qontinui.extraction.runtime.uitars.hal_dispatch import failsafe_corners

        assert failsafe_corners((0, 0, 800, 600)) == {
            (0, 0),
            (0, 599),
            (799, 0),
            (799, 599),
        }

    def test_unknown_rect_checks_only_origin(self):
        from qontinui.extraction.runtime.uitars.hal_dispatch import failsafe_corners

        assert failsafe_corners(None) == {(0, 0)}

    @pytest.mark.parametrize("platform", ["win32", "darwin"])
    def test_non_x11_uses_the_monitor_at_the_origin(self, platform):
        """Windows SM_CXSCREEN / macOS main display: the monitor at (0, 0),
        even when mss lists another monitor first."""
        from qontinui.extraction.runtime.uitars.hal_dispatch import (
            failsafe_rect_from_monitors,
        )

        assert failsafe_rect_from_monitors(LEFT_THEN_PRIMARY, platform) == (
            0,
            0,
            2560,
            1440,
        )

    @pytest.mark.parametrize("platform", ["linux", "freebsd14", "openbsd7", "sunos5"])
    def test_x11_uses_the_whole_x_screen(self, platform):
        """X11 width_in_pixels/height_in_pixels anchored at (0, 0), on every
        platform that is neither Windows nor macOS."""
        from qontinui.extraction.runtime.uitars.hal_dispatch import (
            failsafe_rect_from_monitors,
        )

        assert failsafe_rect_from_monitors(X11_TWO_MONITORS, platform) == (
            0,
            0,
            4480,
            1440,
        )

    def test_x11_rect_is_anchored_at_the_origin(self):
        """Monitors not touching (0, 0) still give a (0, 0)-anchored rectangle
        reaching max(x + width), max(y + height), as pyautogui's root size does."""
        from qontinui.extraction.runtime.uitars.hal_dispatch import (
            failsafe_rect_from_monitors,
        )

        monitors = [_monitor(0, 100, 50, 800, 600), _monitor(1, 900, 200, 1024, 768)]
        assert failsafe_rect_from_monitors(monitors, "linux") == (0, 0, 1924, 968)

    def test_no_monitor_at_origin_is_unknown_off_x11(self):
        from qontinui.extraction.runtime.uitars.hal_dispatch import (
            failsafe_rect_from_monitors,
        )

        monitors = [_monitor(0, 100, 100, 800, 600)]
        assert failsafe_rect_from_monitors(monitors, "win32") is None
        assert failsafe_rect_from_monitors([], "linux") is None

    def test_rect_resolved_from_hal_monitors(self, monkeypatch):
        from qontinui.extraction.runtime.uitars import hal_dispatch

        monkeypatch.setattr(hal_dispatch, "_current_platform", lambda: "win32")
        capture = MagicMock()
        capture.get_monitors.return_value = LEFT_THEN_PRIMARY
        with patch(
            "qontinui.hal.factory.HALFactory.get_screen_capture", return_value=capture
        ):
            assert _REAL_RESOLVE_FAILSAFE_RECT() == (0, 0, 2560, 1440)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("position", "aborts"),
        [((2559, 1439), True), ((0, 1439), True), ((-1920, 0), False), ((-1, 1079), False)],
    )
    async def test_multi_monitor_corners_follow_the_primary_on_windows(
        self, monkeypatch, position, aborts
    ):
        """Primary-monitor corners abort; the left monitor's corners do not."""
        from qontinui.extraction.runtime.uitars import hal_dispatch

        monkeypatch.setattr(hal_dispatch, "_current_platform", lambda: "win32")
        monkeypatch.setattr(
            hal_dispatch, "_resolve_failsafe_rect", _REAL_RESOLVE_FAILSAFE_RECT
        )
        capture = MagicMock()
        capture.get_monitors.return_value = LEFT_THEN_PRIMARY
        mouse, keyboard = _make_hal_mocks(position=position)
        with patch(
            "qontinui.hal.factory.HALFactory.get_screen_capture", return_value=capture
        ):
            result = await hal_dispatch.execute_hal_action(
                "click", x=5, y=5, mouse=mouse, keyboard=keyboard
            )

        assert result is (not aborts)
        assert mouse.click_at.called is (not aborts)

    @pytest.mark.parametrize(
        "cause",
        [
            pytest.param("capture_raises", id="screen-capture-raises"),
            pytest.param("no_origin_monitor", id="no-monitor-at-origin"),
        ],
    )
    def test_unavailable_rect_falls_back_and_warns_once(self, monkeypatch, caplog, cause):
        from qontinui.extraction.runtime.uitars import hal_dispatch

        monkeypatch.setattr(hal_dispatch, "_current_platform", lambda: "win32")
        monkeypatch.setattr(hal_dispatch, "_warned_no_failsafe_rect", False)
        if cause == "capture_raises":
            capture_kwargs: dict[str, object] = {"side_effect": RuntimeError("no display")}
        else:
            capture = MagicMock()
            capture.get_monitors.return_value = [_monitor(0, 50, 50, 10, 10)]
            capture_kwargs = {"return_value": capture}
        with (
            patch("qontinui.hal.factory.HALFactory.get_screen_capture", **capture_kwargs),
            caplog.at_level("WARNING"),
        ):
            assert _REAL_RESOLVE_FAILSAFE_RECT() is None
            assert _REAL_RESOLVE_FAILSAFE_RECT() is None

        warnings = [
            r for r in caplog.records if "Failsafe screen rectangle unavailable" in r.message
        ]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_without_rect_only_origin_aborts(self, monkeypatch):
        from qontinui.extraction.runtime.uitars import hal_dispatch

        monkeypatch.setattr(hal_dispatch, "_resolve_failsafe_rect", lambda: None)
        mouse, keyboard = _make_hal_mocks(position=(1919, 1079))
        assert (
            await hal_dispatch.execute_hal_action(
                "click", x=1, y=1, mouse=mouse, keyboard=keyboard
            )
            is True
        )
        mouse.get_mouse_position.return_value = MousePosition(0, 0)
        assert (
            await hal_dispatch.execute_hal_action(
                "click", x=1, y=1, mouse=mouse, keyboard=keyboard
            )
            is False
        )

    @pytest.mark.asyncio
    async def test_explicit_rect_overrides_resolution(self):
        from qontinui.extraction.runtime.uitars import hal_dispatch

        mouse, keyboard = _make_hal_mocks(position=(639, 479))
        result = await hal_dispatch.execute_hal_action(
            "click", x=1, y=1, mouse=mouse, keyboard=keyboard, failsafe_rect=(0, 0, 640, 480)
        )
        assert result is False
        mouse.click_at.assert_not_called()


# Every UI-TARS action that performs input, with the inputs it needs.
INPUT_ACTIONS = [
    ("click", {"x": 1, "y": 2}),
    ("double_click", {"x": 1, "y": 2}),
    ("right_click", {"x": 1, "y": 2}),
    ("type", {"x": 1, "y": 2, "text": "abc"}),
    ("type", {"text": "abc"}),
    ("scroll", {"scroll_direction": "up", "scroll_amount": 3}),
    ("scroll", {"scroll_direction": "down", "scroll_amount": 3}),
    ("scroll", {"scroll_direction": "left", "scroll_amount": 3}),
    ("scroll", {"scroll_direction": "right", "scroll_amount": 3}),
    ("hover", {"x": 1, "y": 2}),
    ("drag", {"x": 1, "y": 2, "end_x": 3, "end_y": 4}),
    ("hotkey", {"keys": ["ctrl", "c"]}),
]
INPUT_METHODS = (
    "click_at",
    "double_click_at",
    "move_mouse",
    "drag",
    "mouse_scroll",
    "mouse_move",
    "mouse_click",
    "mouse_down",
    "mouse_up",
    "mouse_drag",
)


class TestFailsafeCoversEveryInputAction:
    """Every input-performing action is refused on a corner, touching nothing."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("action_type", "kwargs"), INPUT_ACTIONS)
    @pytest.mark.parametrize("corner", [(0, 0), (1919, 1079)])
    async def test_refused_on_corner(self, action_type, kwargs, corner):
        from qontinui.extraction.runtime.uitars.hal_dispatch import execute_hal_action

        mouse, keyboard = _make_hal_mocks(position=corner)

        result = await execute_hal_action(
            action_type, **kwargs, mouse=mouse, keyboard=keyboard
        )

        assert result is False
        for name in INPUT_METHODS:
            getattr(mouse, name).assert_not_called()
        assert keyboard.mock_calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("action_type", "kwargs"), INPUT_ACTIONS)
    async def test_executed_off_corner(self, action_type, kwargs):
        """Control: the same actions run when the pointer is not on a corner."""
        from qontinui.extraction.runtime.uitars.hal_dispatch import execute_hal_action

        mouse, keyboard = _make_hal_mocks(position=(500, 500))

        assert (
            await execute_hal_action(action_type, **kwargs, mouse=mouse, keyboard=keyboard)
            is True
        )
