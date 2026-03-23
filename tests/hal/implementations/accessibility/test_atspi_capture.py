"""Tests for the AT-SPI Linux accessibility capture backend.

All AT-SPI / GI imports are mocked so the tests run on any platform.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from qontinui_schemas.accessibility import (
    AccessibilityConfig,
    AccessibilityRole,
)

from qontinui.hal.implementations.accessibility.atspi_capture import (
    ATSPI_ROLE_MAP,
    ATSPIAccessibilityCapture,
)

# ---------------------------------------------------------------------------
# Helpers to build mock AT-SPI accessible objects
# ---------------------------------------------------------------------------


def _mock_atspi_element(
    name: str = "",
    role_name: str = "push button",
    extents: tuple[int, int, int, int] | None = (10, 20, 100, 30),
    states: set[str] | None = None,
    children: list | None = None,
    value: str | None = None,
    description: str | None = None,
    automation_id: str | None = None,
) -> MagicMock:
    """Create a mock AT-SPI accessible element.

    The mock exposes the *pyatspi2* (non-GI) API surface by default,
    which is what the capture uses when ``_use_gi`` is False.
    """
    if states is None:
        states = ["enabled", "sensitive", "showing", "visible"]
    if children is None:
        children = []

    el = MagicMock()

    # Name / description
    el.name = name
    el.description = description or ""

    # Role
    el.getRoleName.return_value = role_name

    # Children
    el.childCount = len(children)
    el.getChildAtIndex.side_effect = lambda i: children[i] if i < len(children) else None

    # Extents (component interface)
    if extents is not None:
        comp = MagicMock()
        rect = MagicMock()
        rect.x, rect.y, rect.width, rect.height = extents
        comp.getExtents.return_value = rect
        el.queryComponent.return_value = comp
    else:
        el.queryComponent.side_effect = Exception("no component")

    # States
    state_set = MagicMock()
    state_set.getStates.return_value = list(range(len(states)))
    el.getState.return_value = state_set

    # Value / text interfaces – default to not available
    el.queryText.side_effect = Exception("no text")
    el.queryValue.side_effect = Exception("no value")
    if value is not None:
        text_iface = MagicMock()
        text_iface.characterCount = len(value)
        text_iface.getText.return_value = value
        el.queryText.side_effect = None
        el.queryText.return_value = text_iface

    # Automation id (pyatspi2 path – attribute style)
    el.id = automation_id

    return el


def _make_capture(
    use_gi: bool = False,
    config: AccessibilityConfig | None = None,
) -> ATSPIAccessibilityCapture:
    """Build an ATSPIAccessibilityCapture with AT-SPI mocked out."""
    capture = ATSPIAccessibilityCapture(config=config)
    # Simulate a successful import of pyatspi
    capture._atspi = MagicMock()
    capture._use_gi = use_gi
    # Set DESKTOP_COORDS constant used by non-GI path
    capture._atspi.DESKTOP_COORDS = 0
    # Provide stateToString so _get_states works for pyatspi2 path
    _state_names = ["enabled", "sensitive", "showing", "visible"]
    capture._atspi.stateToString = lambda idx: (
        _state_names[idx] if idx < len(_state_names) else None
    )
    return capture


# ===================================================================
# 1. Role mapping tests
# ===================================================================


class TestAtspiRoleMap:
    """Tests for the ATSPI_ROLE_MAP constant."""

    def test_atspi_role_map_completeness(self):
        """Critical UI roles must be present in the role map."""
        critical_keys = [
            "push button",
            "text",
            "combo box",
            "check box",
            "radio button",
            "menu",
            "menu item",
            "link",
            "list",
            "list item",
            "tab",
            "slider",
            "dialog",
            "toolbar",
            "table",
            "heading",
            "image",
            "label",
            "entry",
            "tree",
        ]
        for key in critical_keys:
            assert key in ATSPI_ROLE_MAP, f"Missing critical role: {key}"

    def test_atspi_role_map_values(self):
        """Spot-check specific role name → AccessibilityRole mappings."""
        assert ATSPI_ROLE_MAP["push button"] == AccessibilityRole.BUTTON
        assert ATSPI_ROLE_MAP["text"] == AccessibilityRole.TEXTBOX
        assert ATSPI_ROLE_MAP["combo box"] == AccessibilityRole.COMBOBOX
        assert ATSPI_ROLE_MAP["check box"] == AccessibilityRole.CHECKBOX
        assert ATSPI_ROLE_MAP["radio button"] == AccessibilityRole.RADIO
        assert ATSPI_ROLE_MAP["menu"] == AccessibilityRole.MENU
        assert ATSPI_ROLE_MAP["menu item"] == AccessibilityRole.MENUITEM
        assert ATSPI_ROLE_MAP["link"] == AccessibilityRole.LINK
        assert ATSPI_ROLE_MAP["label"] == AccessibilityRole.STATIC_TEXT
        assert ATSPI_ROLE_MAP["entry"] == AccessibilityRole.EDIT
        assert ATSPI_ROLE_MAP["frame"] == AccessibilityRole.WINDOW
        assert ATSPI_ROLE_MAP["panel"] == AccessibilityRole.GROUP
        assert ATSPI_ROLE_MAP["toggle button"] == AccessibilityRole.SWITCH
        assert ATSPI_ROLE_MAP["password text"] == AccessibilityRole.TEXTBOX
        assert ATSPI_ROLE_MAP["heading"] == AccessibilityRole.HEADING
        assert ATSPI_ROLE_MAP["image"] == AccessibilityRole.IMG


# ===================================================================
# 2. Element-to-node conversion tests
# ===================================================================


class TestElementToNode:
    """Tests for ATSPIAccessibilityCapture._element_to_node."""

    def test_element_to_node_basic(self):
        """A mock AT-SPI element is converted to the correct AccessibilityNode."""
        capture = _make_capture()
        config = AccessibilityConfig()

        element = _mock_atspi_element(
            name="Save",
            role_name="push button",
            extents=(100, 200, 80, 30),
        )

        node = capture._element_to_node(element, config, depth=0)

        assert node is not None
        assert node.name == "Save"
        assert node.role == AccessibilityRole.BUTTON
        assert node.bounds is not None
        assert node.bounds.x == 100
        assert node.bounds.y == 200
        assert node.bounds.width == 80
        assert node.bounds.height == 30
        assert node.is_interactive is True

    def test_element_to_node_with_children(self):
        """Parent element's children are converted into the node tree."""
        capture = _make_capture()
        config = AccessibilityConfig()

        child1 = _mock_atspi_element(name="OK", role_name="push button")
        child2 = _mock_atspi_element(name="Cancel", role_name="push button")

        parent = _mock_atspi_element(
            name="Toolbar",
            role_name="toolbar",
            children=[child1, child2],
        )

        node = capture._element_to_node(parent, config, depth=0)

        assert node is not None
        assert len(node.children) == 2
        assert node.children[0].name == "OK"
        assert node.children[0].role == AccessibilityRole.BUTTON
        assert node.children[1].name == "Cancel"

    def test_element_to_node_skips_hidden(self):
        """An element that is not showing/visible is marked hidden in state."""
        capture = _make_capture()
        config = AccessibilityConfig()

        # Override _get_states to return no showing/visible flags
        hidden_element = _mock_atspi_element(
            name="Secret",
            role_name="push button",
            states=["enabled", "sensitive"],  # missing 'showing' and 'visible'
        )

        node = capture._element_to_node(hidden_element, config, depth=0)

        assert node is not None
        assert node.state is not None
        assert node.state.is_hidden is True

    def test_element_to_node_respects_max_depth(self):
        """Elements beyond max_depth are not converted."""
        capture = _make_capture()
        config = AccessibilityConfig(max_depth=0)

        element = _mock_atspi_element(name="Deep", role_name="push button")

        # depth=0 is allowed by max_depth=0
        node_at_0 = capture._element_to_node(element, config, depth=0)
        assert node_at_0 is not None

        # depth=1 exceeds max_depth=0 → should be None
        node_at_1 = capture._element_to_node(element, config, depth=1)
        assert node_at_1 is None

    def test_element_to_node_registers_in_ref_manager(self):
        """Converted nodes are registered with the ref manager for persistence."""
        capture = _make_capture()
        config = AccessibilityConfig()

        element = _mock_atspi_element(
            name="Save",
            role_name="push button",
        )

        node = capture._element_to_node(element, config, depth=0)
        assert node is not None

        # The ref manager should have the node registered
        found = capture._ref_manager.get_node_by_ref(node.ref)
        assert found is not None
        assert found.name == "Save"

    def test_element_to_node_text_value(self):
        """Text interface value is captured."""
        capture = _make_capture()
        config = AccessibilityConfig()

        element = _mock_atspi_element(
            name="Search",
            role_name="entry",
            value="hello world",
        )

        node = capture._element_to_node(element, config, depth=0)

        assert node is not None
        assert node.value == "hello world"
        assert node.role == AccessibilityRole.EDIT

    def test_element_to_node_unknown_role(self):
        """An unrecognized role name falls back to GENERIC."""
        capture = _make_capture()
        config = AccessibilityConfig()

        element = _mock_atspi_element(
            name="Widget",
            role_name="some totally made up role",
        )

        node = capture._element_to_node(element, config, depth=0)

        assert node is not None
        assert node.role == AccessibilityRole.GENERIC


# ===================================================================
# 3. Ref element mapping tests
# ===================================================================


class TestFindAtspiElementByRef:
    """Tests for _find_atspi_element_by_ref."""

    def test_find_atspi_element_by_ref(self):
        """After _element_to_node, the live AT-SPI element is retrievable by ref."""
        capture = _make_capture()
        config = AccessibilityConfig()

        element = _mock_atspi_element(name="Open", role_name="push button")
        node = capture._element_to_node(element, config, depth=0)
        assert node is not None

        found_element = capture._find_atspi_element_by_ref(node.ref)
        assert found_element is element

    def test_find_atspi_element_by_ref_missing(self):
        """Returns None for an unknown ref."""
        capture = _make_capture()

        result = capture._find_atspi_element_by_ref("@nonexistent")
        assert result is None

    def test_find_atspi_element_by_ref_multiple(self):
        """Each child's AT-SPI element is independently retrievable."""
        capture = _make_capture()
        config = AccessibilityConfig()

        child1 = _mock_atspi_element(name="A", role_name="push button")
        child2 = _mock_atspi_element(name="B", role_name="push button")
        parent = _mock_atspi_element(
            name="Container",
            role_name="panel",
            children=[child1, child2],
        )

        parent_node = capture._element_to_node(parent, config, depth=0)
        assert parent_node is not None

        # All three refs should be mapped
        assert capture._find_atspi_element_by_ref(parent_node.ref) is parent
        assert capture._find_atspi_element_by_ref(parent_node.children[0].ref) is child1
        assert capture._find_atspi_element_by_ref(parent_node.children[1].ref) is child2


# ===================================================================
# 4. Ref persistence tests
# ===================================================================


class TestAtspiRefPersistence:
    """Tests for ref save/load across disconnect/reconnect."""

    def test_atspi_disconnect_saves_refs(self, tmp_path: Path):
        """disconnect() calls save() on the ref manager when refs exist."""
        import asyncio

        capture = _make_capture()
        capture._is_connected = True
        capture._persistence_dir = tmp_path
        config = AccessibilityConfig()

        # Set a target element so _ref_persistence_path returns a path
        target = _mock_atspi_element(name="TestApp", role_name="frame")
        capture._target_element = target

        # Build a node so the ref manager is non-empty
        btn = _mock_atspi_element(name="OK", role_name="push button")
        capture._element_to_node(btn, config, depth=0)

        assert capture._ref_manager.count > 0

        # Disconnect should persist
        asyncio.get_event_loop().run_until_complete(capture.disconnect())

        # A file should have been written
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1

    def test_atspi_capture_restores_refs(self, tmp_path: Path):
        """Persisted refs are re-resolved against a new capture tree."""
        import asyncio
        import json

        # Write a fake fingerprint file
        fp_data = {
            "@e1": {
                "automation_id": "btnSave",
                "role": "button",
                "name": "Save",
            }
        }
        fp_file = tmp_path / "TestApp.json"
        fp_file.write_text(json.dumps(fp_data), encoding="utf-8")

        capture = _make_capture()
        capture._is_connected = True
        capture._persistence_dir = tmp_path
        capture._atspi_elements_by_ref.clear()

        # Create target element
        target = _mock_atspi_element(name="TestApp", role_name="frame")
        capture._target_element = target

        # Build a tree that contains a matching node (automation_id = btnSave)
        btn = _mock_atspi_element(
            name="Save",
            role_name="push button",
            automation_id="btnSave",
        )
        frame = _mock_atspi_element(
            name="TestApp",
            role_name="frame",
            children=[btn],
        )
        capture._target_element = frame

        snapshot = asyncio.get_event_loop().run_until_complete(capture.capture_tree())

        # The restored ref should be findable
        restored_node = capture._ref_manager.get_node_by_ref("@e1")
        assert restored_node is not None
        assert restored_node.name == "Save"


# ===================================================================
# 5. Platform guard tests
# ===================================================================


class TestPlatformGuard:
    """Tests for platform-specific behaviour."""

    def test_atspi_connect_fails_on_non_linux(self):
        """connect() returns False when not on Linux."""
        import asyncio

        with patch(
            "qontinui.hal.implementations.accessibility.atspi_capture._is_linux",
            return_value=False,
        ):
            capture = ATSPIAccessibilityCapture()
            # _atspi is None, _ensure_atspi_available will check _is_linux
            result = asyncio.get_event_loop().run_until_complete(capture.connect(target="anything"))
            assert result is False

    def test_atspi_ensure_available_returns_true_when_already_loaded(self):
        """_ensure_atspi_available returns True if _atspi is already set."""
        capture = _make_capture()
        assert capture._ensure_atspi_available() is True

    def test_is_connected_false_by_default(self):
        """A fresh capture instance is not connected."""
        capture = ATSPIAccessibilityCapture()
        assert capture.is_connected() is False

    def test_get_backend_name(self):
        """Backend name is 'atspi'."""
        capture = ATSPIAccessibilityCapture()
        assert capture.get_backend_name() == "atspi"
