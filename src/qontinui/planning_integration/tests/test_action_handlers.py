"""Tests for planning_integration action handlers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from multistate.planning.blackboard import Blackboard

from qontinui.discovery.target_connection import BoundingBox, Element
from qontinui.planning_integration.action_handlers import create_action_handlers


@pytest.fixture()
def hal() -> MagicMock:
    """Mock HAL container with keyboard and mouse controllers."""
    mock = MagicMock()
    mock.mouse_controller.mouse_click = MagicMock()
    mock.keyboard_controller.type_text = MagicMock()
    mock.keyboard_controller.key_press = MagicMock()
    mock.keyboard_controller.hotkey = MagicMock()
    return mock


@pytest.fixture()
def sample_element() -> Element:
    """Element with a bounding box."""
    return Element(
        id="btn-submit",
        tag_name="button",
        text_content="Submit",
        bbox=BoundingBox(x=100.0, y=200.0, width=80.0, height=30.0),
        is_visible=True,
        is_enabled=True,
    )


@pytest.fixture()
def ui_connection(sample_element: Element) -> AsyncMock:
    """Mock async UI Bridge connection."""
    conn = AsyncMock()
    conn.find_elements = AsyncMock(return_value=[sample_element])
    return conn


@pytest.fixture()
def state_manager() -> MagicMock:
    """Mock state manager."""
    mgr = MagicMock()
    mgr.navigate_to = MagicMock(return_value=True)
    mgr.execute_transition = MagicMock(return_value=True)
    mgr.get_active_states = MagicMock(return_value={"main_page"})
    return mgr


@pytest.fixture()
def blackboard(ui_connection: AsyncMock, state_manager: MagicMock) -> Blackboard:
    """Blackboard pre-populated with UI connection and state manager."""
    bb = Blackboard()
    bb.set("_ui_connection", ui_connection)
    bb.set("_state_manager", state_manager)
    return bb


@pytest.fixture()
def handlers(
    hal: MagicMock,
    ui_connection: AsyncMock,
    state_manager: MagicMock,
) -> dict:
    """Action handlers dict from create_action_handlers."""
    return create_action_handlers(
        hal=hal,
        ui_connection=ui_connection,
        state_manager=state_manager,
    )


class TestClickElement:
    """Tests for click_element handler."""

    def test_click_element_with_bbox(
        self,
        handlers: dict,
        blackboard: Blackboard,
        hal: MagicMock,
    ) -> None:
        """Click calls mouse_click at bbox center coordinates."""
        handlers["click_element"](blackboard, "btn-submit")

        # BoundingBox(100, 200, 80, 30) -> center = (140, 215)
        hal.mouse_controller.mouse_click.assert_called_once_with(140, 215)

    def test_click_element_not_found(
        self,
        handlers: dict,
        blackboard: Blackboard,
        ui_connection: AsyncMock,
    ) -> None:
        """RuntimeError raised when element ID is not in the bridge."""
        ui_connection.find_elements = AsyncMock(return_value=[])

        with pytest.raises(RuntimeError, match="not found"):
            handlers["click_element"](blackboard, "nonexistent-btn")

    def test_click_element_no_bbox(
        self,
        handlers: dict,
        blackboard: Blackboard,
        ui_connection: AsyncMock,
    ) -> None:
        """RuntimeError raised when element has no bounding box."""
        no_bbox_elem = Element(id="no-bbox", tag_name="span", bbox=None, is_visible=True)
        ui_connection.find_elements = AsyncMock(return_value=[no_bbox_elem])

        with pytest.raises(RuntimeError, match="no bounding box"):
            handlers["click_element"](blackboard, "no-bbox")


class TestTypeText:
    """Tests for type_text handler."""

    def test_type_text_keyboard_direct(
        self,
        handlers: dict,
        blackboard: Blackboard,
        hal: MagicMock,
    ) -> None:
        """_keyboard element_id types directly without clicking."""
        handlers["type_text"](blackboard, "_keyboard", "hello")

        hal.keyboard_controller.type_text.assert_called_once_with("hello")
        hal.mouse_controller.mouse_click.assert_not_called()

    def test_type_text_on_element(
        self,
        handlers: dict,
        blackboard: Blackboard,
        hal: MagicMock,
    ) -> None:
        """Typing on an element clicks it first, then types."""
        handlers["type_text"](blackboard, "btn-submit", "some text")

        # Should have clicked to focus
        hal.mouse_controller.mouse_click.assert_called_once_with(140, 215)
        # Then typed
        hal.keyboard_controller.type_text.assert_called_once_with("some text")


class TestNavigatePath:
    """Tests for navigate_path handler."""

    def test_navigate_path_success(
        self,
        handlers: dict,
        blackboard: Blackboard,
        state_manager: MagicMock,
    ) -> None:
        """navigate_path calls state_manager.navigate_to with target."""
        handlers["navigate_path"](blackboard, "settings_page")

        state_manager.navigate_to.assert_called_once_with(["settings_page"])

    def test_navigate_path_failure(
        self,
        handlers: dict,
        blackboard: Blackboard,
        state_manager: MagicMock,
    ) -> None:
        """RuntimeError raised when navigation fails."""
        state_manager.navigate_to.return_value = False

        with pytest.raises(RuntimeError, match="could not reach"):
            handlers["navigate_path"](blackboard, "unreachable_page")

    def test_navigate_path_no_manager(
        self,
        hal: MagicMock,
    ) -> None:
        """RuntimeError raised when no state manager is available."""
        handlers = create_action_handlers(hal=hal)
        bb = Blackboard()

        with pytest.raises(RuntimeError, match="No state manager"):
            handlers["navigate_path"](bb, "any_state")


class TestNavigateTransition:
    """Tests for navigate_transition handler."""

    def test_navigate_transition_success(
        self,
        handlers: dict,
        blackboard: Blackboard,
        state_manager: MagicMock,
    ) -> None:
        """execute_transition called with the transition ID."""
        handlers["navigate_transition"](blackboard, "t_main_to_settings")

        state_manager.execute_transition.assert_called_once_with("t_main_to_settings")

    def test_navigate_transition_failure(
        self,
        handlers: dict,
        blackboard: Blackboard,
        state_manager: MagicMock,
    ) -> None:
        """RuntimeError raised when transition fails."""
        state_manager.execute_transition.return_value = False

        with pytest.raises(RuntimeError, match="did not succeed"):
            handlers["navigate_transition"](blackboard, "bad_transition")


class TestWaitForState:
    """Tests for wait_for_state handler."""

    def test_wait_for_state_immediate(
        self,
        handlers: dict,
        blackboard: Blackboard,
        state_manager: MagicMock,
    ) -> None:
        """Returns immediately when state is already active."""
        state_manager.get_active_states.return_value = {"target_state"}

        handlers["wait_for_state"](blackboard, "target_state")

        state_manager.get_active_states.assert_called_once()

    @patch("qontinui.planning_integration.action_handlers.time")
    def test_wait_for_state_timeout(
        self,
        mock_time: MagicMock,
        handlers: dict,
        blackboard: Blackboard,
        state_manager: MagicMock,
    ) -> None:
        """RuntimeError raised when state never appears."""
        mock_time.sleep = MagicMock()
        state_manager.get_active_states.return_value = {"other_state"}

        with pytest.raises(RuntimeError, match="Timed out"):
            handlers["wait_for_state"](blackboard, "missing_state")


class TestWaitForElement:
    """Tests for wait_for_element handler."""

    @patch("qontinui.planning_integration.action_handlers.time")
    def test_wait_for_element_timeout(
        self,
        mock_time: MagicMock,
        handlers: dict,
        blackboard: Blackboard,
        ui_connection: AsyncMock,
    ) -> None:
        """RuntimeError raised when element never becomes visible."""
        mock_time.sleep = MagicMock()
        ui_connection.find_elements = AsyncMock(return_value=[])

        with pytest.raises(RuntimeError, match="Timed out"):
            handlers["wait_for_element"](blackboard, "ghost-element")

    def test_wait_for_element_found(
        self,
        handlers: dict,
        blackboard: Blackboard,
        sample_element: Element,
    ) -> None:
        """Returns immediately when element is already visible."""
        handlers["wait_for_element"](blackboard, "btn-submit")


class TestDismissDialog:
    """Tests for dismiss_dialog handler."""

    @patch("qontinui.planning_integration.action_handlers.time")
    def test_dismiss_dialog(
        self,
        mock_time: MagicMock,
        handlers: dict,
        blackboard: Blackboard,
        hal: MagicMock,
        state_manager: MagicMock,
    ) -> None:
        """Escape is pressed and dialog state is no longer active."""
        mock_time.sleep = MagicMock()
        state_manager.get_active_states.return_value = {"main_page"}

        handlers["dismiss_dialog"](blackboard, "dialog_confirm")

        hal.keyboard_controller.key_press.assert_called_once_with("escape")

    @patch("qontinui.planning_integration.action_handlers.time")
    def test_dismiss_dialog_still_active(
        self,
        mock_time: MagicMock,
        handlers: dict,
        blackboard: Blackboard,
        hal: MagicMock,
        state_manager: MagicMock,
    ) -> None:
        """RuntimeError raised when dialog state persists after Escape."""
        mock_time.sleep = MagicMock()
        state_manager.get_active_states.return_value = {"dialog_confirm", "main_page"}

        with pytest.raises(RuntimeError, match="still active"):
            handlers["dismiss_dialog"](blackboard, "dialog_confirm")
