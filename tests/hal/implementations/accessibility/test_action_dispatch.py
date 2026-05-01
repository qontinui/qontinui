"""Tests for control-type-aware action dispatch registry."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from qontinui_schemas.accessibility import AccessibilityBounds, AccessibilityNode, AccessibilityRole

from qontinui.hal.implementations.accessibility.action_dispatch import (
    _POST_ACTION_WAIT_S,
    ActionDispatchRegistry,
    ButtonStrategy,
    CheckboxStrategy,
    ComboBoxStrategy,
    MenuItemStrategy,
    SliderStrategy,
    TextBoxStrategy,
    TreeItemStrategy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(
    ref: str = "@e1",
    role: AccessibilityRole = AccessibilityRole.BUTTON,
    name: str | None = None,
    value: str | None = None,
    automation_id: str | None = None,
    is_interactive: bool = True,
    bounds: AccessibilityBounds | None = None,
    children: list[AccessibilityNode] | None = None,
) -> AccessibilityNode:
    """Build an AccessibilityNode for testing."""
    if bounds is None:
        bounds = AccessibilityBounds(x=100, y=200, width=80, height=30)
    return AccessibilityNode(
        ref=ref,
        role=role,
        name=name,
        value=value,
        automation_id=automation_id,
        is_interactive=is_interactive,
        bounds=bounds,
        children=children or [],
    )


def _make_capture(
    perform_action_return: bool = True,
    click_by_ref_return: bool = True,
    type_by_ref_return: bool = True,
) -> MagicMock:
    """Build a mock IAccessibilityCapture."""
    capture = MagicMock()
    capture.perform_action = AsyncMock(return_value=perform_action_return)
    capture.click_by_ref = AsyncMock(return_value=click_by_ref_return)
    capture.type_by_ref = AsyncMock(return_value=type_by_ref_return)
    return capture


# ---------------------------------------------------------------------------
# TestActionDispatchRegistry
# ---------------------------------------------------------------------------


class TestActionDispatchRegistry:
    def test_registered_roles_have_strategies(self):
        """All expected roles have registered strategies."""
        registry = ActionDispatchRegistry()
        for role in [
            AccessibilityRole.BUTTON,
            AccessibilityRole.COMBOBOX,
            AccessibilityRole.TREEITEM,
            AccessibilityRole.MENUITEM,
            AccessibilityRole.CHECKBOX,
            AccessibilityRole.SLIDER,
            AccessibilityRole.TEXTBOX,
        ]:
            assert (
                registry.get_strategy(role) is not None
            ), f"Missing strategy for {role}"

    def test_unregistered_role_returns_none(self):
        """Roles without a strategy return None."""
        registry = ActionDispatchRegistry()
        assert registry.get_strategy(AccessibilityRole.WINDOW) is None

    def test_speed_profile_fast(self):
        """Fast profile halves wait times."""
        registry = ActionDispatchRegistry(speed_profile="fast")
        assert registry._speed_multiplier == 0.5

    def test_speed_profile_slow(self):
        """Slow profile triples wait times."""
        registry = ActionDispatchRegistry(speed_profile="slow")
        assert registry._speed_multiplier == 3.0

    def test_speed_profile_default(self):
        """Default profile uses multiplier 1.0."""
        registry = ActionDispatchRegistry(speed_profile="default")
        assert registry._speed_multiplier == 1.0

    def test_speed_profile_unknown_defaults_to_one(self):
        """Unknown profile falls back to multiplier 1.0."""
        registry = ActionDispatchRegistry(speed_profile="turbo")
        assert registry._speed_multiplier == 1.0

    def test_register_replaces_strategy(self):
        """Registering a new strategy for an existing role replaces it."""
        registry = ActionDispatchRegistry()
        new_strategy = ButtonStrategy()
        registry.register(AccessibilityRole.CHECKBOX, new_strategy)
        assert registry.get_strategy(AccessibilityRole.CHECKBOX) is new_strategy

    def test_menuitemcheckbox_has_strategy(self):
        """MENUITEMCHECKBOX is also registered."""
        registry = ActionDispatchRegistry()
        assert registry.get_strategy(AccessibilityRole.MENUITEMCHECKBOX) is not None

    def test_menuitemradio_has_strategy(self):
        """MENUITEMRADIO is also registered."""
        registry = ActionDispatchRegistry()
        assert registry.get_strategy(AccessibilityRole.MENUITEMRADIO) is not None

    def test_edit_has_strategy(self):
        """EDIT role maps to a strategy."""
        registry = ActionDispatchRegistry()
        assert registry.get_strategy(AccessibilityRole.EDIT) is not None

    def test_searchbox_has_strategy(self):
        """SEARCHBOX role maps to a strategy."""
        registry = ActionDispatchRegistry()
        assert registry.get_strategy(AccessibilityRole.SEARCHBOX) is not None


# ---------------------------------------------------------------------------
# TestButtonStrategy
# ---------------------------------------------------------------------------


class TestButtonStrategy:
    def test_button_invoke_success(self):
        """ButtonStrategy tries invoke first."""
        node = _make_node(ref="@btn", role=AccessibilityRole.BUTTON, name="OK")
        capture = _make_capture(perform_action_return=True)

        strategy = ButtonStrategy()
        result = asyncio.run(strategy.execute(node, "click", capture))

        assert result.success
        assert result.action_performed == "invoke"
        capture.perform_action.assert_called_once_with("@btn", "invoke")

    def test_button_fallback_to_click(self):
        """When invoke fails, falls back to click_center."""
        node = _make_node(ref="@btn", role=AccessibilityRole.BUTTON, name="OK")
        capture = _make_capture(perform_action_return=False, click_by_ref_return=True)

        strategy = ButtonStrategy()
        result = asyncio.run(strategy.execute(node, "click", capture))

        assert result.success
        assert result.action_performed == "click_center"
        capture.click_by_ref.assert_called_once_with("@btn")

    def test_button_fallback_click_fails(self):
        """When both invoke and click_center fail, returns failure."""
        node = _make_node(ref="@btn", role=AccessibilityRole.BUTTON, name="OK")
        capture = _make_capture(perform_action_return=False, click_by_ref_return=False)

        strategy = ButtonStrategy()
        result = asyncio.run(strategy.execute(node, "click", capture))

        assert not result.success
        assert result.action_performed == "click_center"

    def test_button_supported_actions(self):
        """ButtonStrategy advertises click, invoke, press."""
        strategy = ButtonStrategy()
        actions = strategy.supported_actions()
        assert "click" in actions
        assert "invoke" in actions
        assert "press" in actions

    def test_button_wait_after_s_is_set(self):
        """Successful invoke returns the expected base wait for BUTTON."""
        node = _make_node(ref="@btn", role=AccessibilityRole.BUTTON, name="OK")
        capture = _make_capture(perform_action_return=True)

        strategy = ButtonStrategy()
        result = asyncio.run(strategy.execute(node, "click", capture))

        assert result.wait_after_s == _POST_ACTION_WAIT_S[AccessibilityRole.BUTTON]


# ---------------------------------------------------------------------------
# TestCheckboxStrategy
# ---------------------------------------------------------------------------


class TestCheckboxStrategy:
    def test_checkbox_toggle_success(self):
        """CheckboxStrategy tries toggle first."""
        node = _make_node(ref="@chk", role=AccessibilityRole.CHECKBOX, name="Agree")
        capture = _make_capture(perform_action_return=True)

        strategy = CheckboxStrategy()
        result = asyncio.run(strategy.execute(node, "toggle", capture))

        assert result.success
        assert result.action_performed == "toggle"
        capture.perform_action.assert_called_once_with("@chk", "toggle")

    def test_checkbox_fallback_to_click(self):
        """When toggle fails, falls back to click_center."""
        node = _make_node(ref="@chk", role=AccessibilityRole.CHECKBOX, name="Agree")
        capture = _make_capture(perform_action_return=False, click_by_ref_return=True)

        strategy = CheckboxStrategy()
        result = asyncio.run(strategy.execute(node, "toggle", capture))

        assert result.success
        assert result.action_performed == "click_center"

    def test_checkbox_supported_actions(self):
        """CheckboxStrategy advertises click, toggle, check, uncheck."""
        strategy = CheckboxStrategy()
        actions = strategy.supported_actions()
        assert "toggle" in actions
        assert "click" in actions
        assert "check" in actions
        assert "uncheck" in actions


# ---------------------------------------------------------------------------
# TestComboBoxStrategy
# ---------------------------------------------------------------------------


class TestComboBoxStrategy:
    def test_combobox_expand_collapse_success(self):
        """ComboBoxStrategy tries expand_collapse first."""
        node = _make_node(ref="@cb", role=AccessibilityRole.COMBOBOX, name="Color")
        capture = _make_capture(perform_action_return=True)

        strategy = ComboBoxStrategy()
        result = asyncio.run(strategy.execute(node, "click", capture))

        assert result.success
        assert result.action_performed == "expand_collapse"

    def test_combobox_fallback_to_click(self):
        """When expand_collapse fails, falls back to click_open."""
        node = _make_node(ref="@cb", role=AccessibilityRole.COMBOBOX, name="Color")
        capture = _make_capture(perform_action_return=False, click_by_ref_return=True)

        strategy = ComboBoxStrategy()
        result = asyncio.run(strategy.execute(node, "click", capture))

        assert result.success
        assert result.action_performed == "click_open"

    def test_combobox_with_item_value(self):
        """ComboBoxStrategy tries to select an item after expand."""
        node = _make_node(ref="@cb", role=AccessibilityRole.COMBOBOX, name="Color")
        # First call (expand_collapse) → True, second call (select_item) → True
        capture = MagicMock()
        capture.perform_action = AsyncMock(return_value=True)

        strategy = ComboBoxStrategy()
        result = asyncio.run(strategy.execute(node, "select", capture, value="Red"))

        assert result.success
        assert result.action_performed == "expand_collapse+select_item"

    def test_combobox_supported_actions(self):
        """ComboBoxStrategy advertises click, expand, collapse, select, open."""
        strategy = ComboBoxStrategy()
        actions = strategy.supported_actions()
        assert "click" in actions
        assert "expand" in actions
        assert "select" in actions


# ---------------------------------------------------------------------------
# TestTreeItemStrategy
# ---------------------------------------------------------------------------


class TestTreeItemStrategy:
    def test_treeitem_select_success(self):
        """TreeItemStrategy tries select first."""
        node = _make_node(ref="@ti", role=AccessibilityRole.TREEITEM, name="Node A")
        capture = _make_capture(perform_action_return=True)

        strategy = TreeItemStrategy()
        result = asyncio.run(strategy.execute(node, "click", capture))

        assert result.success
        assert result.action_performed == "select"

    def test_treeitem_fallback_to_click(self):
        """When select fails, falls back to click_center."""
        node = _make_node(ref="@ti", role=AccessibilityRole.TREEITEM, name="Node A")
        capture = _make_capture(perform_action_return=False, click_by_ref_return=True)

        strategy = TreeItemStrategy()
        result = asyncio.run(strategy.execute(node, "click", capture))

        assert result.success
        assert result.action_performed == "click_center"


# ---------------------------------------------------------------------------
# TestMenuItemStrategy
# ---------------------------------------------------------------------------


class TestMenuItemStrategy:
    def test_menuitem_invoke_success(self):
        """MenuItemStrategy tries invoke first."""
        node = _make_node(ref="@mi", role=AccessibilityRole.MENUITEM, name="File")
        capture = _make_capture(perform_action_return=True)

        strategy = MenuItemStrategy()
        result = asyncio.run(strategy.execute(node, "click", capture))

        assert result.success
        assert result.action_performed == "invoke"

    def test_menuitem_fallback_to_click(self):
        """When invoke fails, falls back to click_center."""
        node = _make_node(ref="@mi", role=AccessibilityRole.MENUITEM, name="File")
        capture = _make_capture(perform_action_return=False, click_by_ref_return=True)

        strategy = MenuItemStrategy()
        result = asyncio.run(strategy.execute(node, "click", capture))

        assert result.success
        assert result.action_performed == "click_center"


# ---------------------------------------------------------------------------
# TestSliderStrategy
# ---------------------------------------------------------------------------


class TestSliderStrategy:
    def test_slider_set_range_value_success(self):
        """SliderStrategy uses set_range_value when value is provided."""
        node = _make_node(ref="@sl", role=AccessibilityRole.SLIDER, name="Volume")
        capture = _make_capture(perform_action_return=True)

        strategy = SliderStrategy()
        result = asyncio.run(strategy.execute(node, "set", capture, value=75))

        assert result.success
        assert result.action_performed == "set_range_value:75"

    def test_slider_fallback_to_click_when_no_value(self):
        """SliderStrategy falls back to click when no value is given."""
        node = _make_node(ref="@sl", role=AccessibilityRole.SLIDER, name="Volume")
        capture = _make_capture(perform_action_return=True, click_by_ref_return=True)

        strategy = SliderStrategy()
        result = asyncio.run(strategy.execute(node, "click", capture))

        assert result.success
        assert result.action_performed == "click_center"

    def test_slider_fallback_to_click_when_set_fails(self):
        """SliderStrategy falls back to click when set_range_value fails."""
        node = _make_node(ref="@sl", role=AccessibilityRole.SLIDER, name="Volume")
        capture = _make_capture(perform_action_return=False, click_by_ref_return=True)

        strategy = SliderStrategy()
        result = asyncio.run(strategy.execute(node, "set", capture, value=50))

        assert result.success
        assert result.action_performed == "click_center"


# ---------------------------------------------------------------------------
# TestTextBoxStrategy
# ---------------------------------------------------------------------------


class TestTextBoxStrategy:
    def test_textbox_set_value_success(self):
        """TextBoxStrategy uses set_value pattern when text is provided."""
        node = _make_node(ref="@tb", role=AccessibilityRole.TEXTBOX, name="Search")
        capture = _make_capture(perform_action_return=True)

        strategy = TextBoxStrategy()
        result = asyncio.run(strategy.execute(node, "type", capture, text="hello"))

        assert result.success
        assert result.action_performed == "set_value:hello"

    def test_textbox_fallback_click_then_type(self):
        """When set_value fails, falls back to click then type."""
        node = _make_node(ref="@tb", role=AccessibilityRole.TEXTBOX, name="Search")
        capture = _make_capture(
            perform_action_return=False,
            click_by_ref_return=True,
            type_by_ref_return=True,
        )

        strategy = TextBoxStrategy()
        result = asyncio.run(strategy.execute(node, "type", capture, text="hello"))

        assert result.success
        assert result.action_performed == "click_then_type"
        capture.type_by_ref.assert_called_once_with("@tb", "hello", clear_first=False)

    def test_textbox_click_focus_no_text(self):
        """With no text, TextBoxStrategy just clicks to focus."""
        node = _make_node(ref="@tb", role=AccessibilityRole.TEXTBOX, name="Search")
        capture = _make_capture(click_by_ref_return=True)

        strategy = TextBoxStrategy()
        result = asyncio.run(strategy.execute(node, "focus", capture))

        assert result.success
        assert result.action_performed == "click_focus"

    def test_textbox_clear_first_passed_through(self):
        """clear_first kwarg is forwarded to type_by_ref."""
        node = _make_node(ref="@tb", role=AccessibilityRole.TEXTBOX, name="Search")
        capture = _make_capture(
            perform_action_return=False,
            click_by_ref_return=True,
            type_by_ref_return=True,
        )

        strategy = TextBoxStrategy()
        asyncio.run(
            strategy.execute(node, "type", capture, text="world", clear_first=True)
        )

        capture.type_by_ref.assert_called_once_with("@tb", "world", clear_first=True)


# ---------------------------------------------------------------------------
# TestDispatchIntegration
# ---------------------------------------------------------------------------


class TestDispatchIntegration:
    def test_dispatch_scales_wait_by_speed(self):
        """dispatch() multiplies wait_after_s by speed multiplier."""
        registry = ActionDispatchRegistry(speed_profile="slow")  # 3x
        node = _make_node(ref="@btn", role=AccessibilityRole.BUTTON, name="OK")
        capture = _make_capture(perform_action_return=True)

        result = asyncio.run(registry.dispatch(node, "click", capture))

        assert result.success
        # Button base wait is 0.09, slow = 3x → 0.27
        expected = _POST_ACTION_WAIT_S[AccessibilityRole.BUTTON] * 3.0
        assert abs(result.wait_after_s - expected) < 0.01

    def test_dispatch_unknown_role_falls_back(self):
        """Unknown role uses default click action."""
        registry = ActionDispatchRegistry()
        node = _make_node(ref="@win", role=AccessibilityRole.WINDOW, name="App")
        capture = _make_capture(click_by_ref_return=True)

        result = asyncio.run(registry.dispatch(node, "click", capture))

        assert result.success
        assert result.action_performed == "click_center"

    def test_dispatch_fast_profile_halves_wait(self):
        """Fast profile halves the base wait."""
        registry = ActionDispatchRegistry(speed_profile="fast")  # 0.5x
        node = _make_node(ref="@chk", role=AccessibilityRole.CHECKBOX, name="Agree")
        capture = _make_capture(perform_action_return=True)

        result = asyncio.run(registry.dispatch(node, "toggle", capture))

        expected = _POST_ACTION_WAIT_S[AccessibilityRole.CHECKBOX] * 0.5
        assert abs(result.wait_after_s - expected) < 0.01

    def test_dispatch_uses_strategy_for_registered_role(self):
        """dispatch() delegates to the correct strategy for a registered role."""
        registry = ActionDispatchRegistry()
        node = _make_node(ref="@cb", role=AccessibilityRole.COMBOBOX, name="Dropdown")
        capture = _make_capture(perform_action_return=True)

        result = asyncio.run(registry.dispatch(node, "click", capture))

        assert result.success
        assert result.action_performed == "expand_collapse"

    def test_dispatch_unknown_role_type_action(self):
        """Unknown role with 'type' action uses default_type fallback."""
        registry = ActionDispatchRegistry()
        node = _make_node(ref="@win", role=AccessibilityRole.WINDOW, name="App")
        capture = _make_capture(type_by_ref_return=True)

        result = asyncio.run(registry.dispatch(node, "type", capture, text="hello"))

        assert result.success
        assert result.action_performed == "default_type"

    def test_dispatch_default_profile_does_not_scale(self):
        """Default profile leaves wait_after_s unchanged (multiplier=1.0)."""
        registry = ActionDispatchRegistry(speed_profile="default")
        node = _make_node(ref="@btn", role=AccessibilityRole.BUTTON, name="OK")
        capture = _make_capture(perform_action_return=True)

        result = asyncio.run(registry.dispatch(node, "click", capture))

        expected = _POST_ACTION_WAIT_S[AccessibilityRole.BUTTON]
        assert abs(result.wait_after_s - expected) < 0.001

    def test_dispatch_kwargs_forwarded_to_strategy(self):
        """Extra kwargs are forwarded through dispatch to the strategy."""
        registry = ActionDispatchRegistry()
        node = _make_node(ref="@tb", role=AccessibilityRole.TEXTBOX, name="Input")
        capture = _make_capture(perform_action_return=True)

        result = asyncio.run(registry.dispatch(node, "type", capture, text="test"))

        assert result.success
        assert result.action_performed == "set_value:test"
