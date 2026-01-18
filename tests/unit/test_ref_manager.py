"""Unit tests for RefManager.

Tests cover:
- Ref assignment to accessibility tree nodes
- Ref lookup by ID
- Interactive-only mode
- Reset functionality
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from qontinui_schemas.accessibility import AccessibilityNode, AccessibilityRole


def _load_module_directly(module_name: str, file_path: Path) -> ModuleType:
    """Load a module directly from file without triggering parent __init__.py.

    This is used to avoid the cv2 import issues in the main qontinui package.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load RefManager directly to avoid cv2 import issues from main qontinui package
# RefManager only depends on qontinui_schemas, so this isolated import works
_ref_manager_path = (
    Path(__file__).parent.parent.parent
    / "src"
    / "qontinui"
    / "hal"
    / "implementations"
    / "accessibility"
    / "ref_manager.py"
)
_ref_manager_module = _load_module_directly(
    "qontinui.hal.implementations.accessibility.ref_manager", _ref_manager_path
)
RefManager = _ref_manager_module.RefManager


class TestRefManagerBasic:
    """Test basic RefManager functionality."""

    def test_empty_init(self) -> None:
        """RefManager initializes with zero count."""
        manager = RefManager()
        assert manager.count == 0
        assert manager.get_all_refs() == []

    def test_assign_single_node(self) -> None:
        """Assigns ref to a single node."""
        node = AccessibilityNode(ref="", role=AccessibilityRole.BUTTON, name="OK")
        manager = RefManager()

        count = manager.assign_refs(node)

        assert count == 1
        assert node.ref == "@e1"
        assert manager.count == 1

    def test_assign_multiple_nodes(self) -> None:
        """Assigns refs to multiple nodes in order."""
        child1 = AccessibilityNode(ref="", role=AccessibilityRole.BUTTON, name="Submit")
        child2 = AccessibilityNode(ref="", role=AccessibilityRole.BUTTON, name="Cancel")
        root = AccessibilityNode(
            ref="",
            role=AccessibilityRole.DIALOG,
            name="Confirm",
            children=[child1, child2],
        )
        manager = RefManager()

        count = manager.assign_refs(root)

        assert count == 3
        assert root.ref == "@e1"
        assert child1.ref == "@e2"
        assert child2.ref == "@e3"

    def test_get_node_by_ref(self) -> None:
        """Gets node by ref ID."""
        child = AccessibilityNode(ref="", role=AccessibilityRole.TEXTBOX, name="Email")
        root = AccessibilityNode(
            ref="", role=AccessibilityRole.FORM, children=[child]
        )
        manager = RefManager()
        manager.assign_refs(root)

        found = manager.get_node_by_ref("@e2")

        assert found is not None
        assert found.name == "Email"
        assert found.role == AccessibilityRole.TEXTBOX

    def test_get_node_by_ref_not_found(self) -> None:
        """Returns None for non-existent ref."""
        node = AccessibilityNode(ref="", role=AccessibilityRole.BUTTON)
        manager = RefManager()
        manager.assign_refs(node)

        found = manager.get_node_by_ref("@e99")

        assert found is None

    def test_get_all_refs(self) -> None:
        """Gets all assigned refs."""
        child1 = AccessibilityNode(ref="", role=AccessibilityRole.BUTTON, name="A")
        child2 = AccessibilityNode(ref="", role=AccessibilityRole.BUTTON, name="B")
        root = AccessibilityNode(ref="", role=AccessibilityRole.GROUP, children=[child1, child2])
        manager = RefManager()
        manager.assign_refs(root)

        refs = manager.get_all_refs()

        assert len(refs) == 3
        assert "@e1" in refs
        assert "@e2" in refs
        assert "@e3" in refs


class TestRefManagerInteractiveOnly:
    """Test interactive-only mode."""

    def test_interactive_only_basic(self) -> None:
        """Only assigns refs to interactive nodes when enabled."""
        interactive = AccessibilityNode(
            ref="", role=AccessibilityRole.BUTTON, name="Click", is_interactive=True
        )
        static = AccessibilityNode(
            ref="", role=AccessibilityRole.STATIC_TEXT, name="Label", is_interactive=False
        )
        root = AccessibilityNode(
            ref="",
            role=AccessibilityRole.DOCUMENT,
            is_interactive=False,
            children=[interactive, static],
        )
        manager = RefManager()

        count = manager.assign_refs(root, interactive_only=True)

        assert count == 1
        assert interactive.ref == "@e1"
        assert static.ref == ""  # Not assigned
        assert root.ref == ""  # Not assigned

    def test_interactive_only_nested(self) -> None:
        """Works correctly with nested structure."""
        button = AccessibilityNode(
            ref="", role=AccessibilityRole.BUTTON, name="Submit", is_interactive=True
        )
        textbox = AccessibilityNode(
            ref="", role=AccessibilityRole.TEXTBOX, name="Input", is_interactive=True
        )
        container = AccessibilityNode(
            ref="",
            role=AccessibilityRole.GROUP,
            is_interactive=False,
            children=[button],
        )
        root = AccessibilityNode(
            ref="",
            role=AccessibilityRole.FORM,
            is_interactive=False,
            children=[container, textbox],
        )
        manager = RefManager()

        count = manager.assign_refs(root, interactive_only=True)

        assert count == 2
        assert button.ref == "@e1"
        assert textbox.ref == "@e2"
        assert container.ref == ""
        assert root.ref == ""

    def test_all_nodes_mode(self) -> None:
        """Assigns refs to all nodes when interactive_only=False."""
        interactive = AccessibilityNode(
            ref="", role=AccessibilityRole.BUTTON, is_interactive=True
        )
        static = AccessibilityNode(
            ref="", role=AccessibilityRole.STATIC_TEXT, is_interactive=False
        )
        root = AccessibilityNode(
            ref="",
            role=AccessibilityRole.DOCUMENT,
            is_interactive=False,
            children=[interactive, static],
        )
        manager = RefManager()

        count = manager.assign_refs(root, interactive_only=False)

        assert count == 3
        assert root.ref == "@e1"
        assert interactive.ref == "@e2"
        assert static.ref == "@e3"


class TestRefManagerReset:
    """Test reset functionality."""

    def test_reset_clears_state(self) -> None:
        """Reset clears all refs and counter."""
        node = AccessibilityNode(ref="", role=AccessibilityRole.BUTTON)
        manager = RefManager()
        manager.assign_refs(node)
        assert manager.count == 1

        manager.reset()

        assert manager.count == 0
        assert manager.get_all_refs() == []
        assert manager.get_node_by_ref("@e1") is None

    def test_assign_refs_calls_reset(self) -> None:
        """assign_refs automatically resets before assigning."""
        node1 = AccessibilityNode(ref="", role=AccessibilityRole.BUTTON)
        node2 = AccessibilityNode(ref="", role=AccessibilityRole.TEXTBOX)
        manager = RefManager()

        manager.assign_refs(node1)
        assert node1.ref == "@e1"

        # Assign to a different tree
        manager.assign_refs(node2)

        assert node2.ref == "@e1"  # Starts from 1 again
        assert manager.count == 1
        # Old ref should not be found
        assert manager.get_node_by_ref("@e1") is node2


class TestRefManagerDeepTree:
    """Test with deeply nested trees."""

    def test_deep_nesting(self) -> None:
        """Handles deeply nested tree structure."""
        # Create a chain of 10 nested nodes
        leaf = AccessibilityNode(ref="", role=AccessibilityRole.BUTTON, name="Leaf")
        current = leaf
        for i in range(9):
            parent = AccessibilityNode(
                ref="",
                role=AccessibilityRole.GROUP,
                name=f"Level{9 - i}",
                children=[current],
            )
            current = parent

        manager = RefManager()
        count = manager.assign_refs(current)

        assert count == 10
        assert current.ref == "@e1"
        assert leaf.ref == "@e10"

    def test_wide_tree(self) -> None:
        """Handles wide tree with many children."""
        children = [
            AccessibilityNode(ref="", role=AccessibilityRole.BUTTON, name=f"Button{i}")
            for i in range(100)
        ]
        root = AccessibilityNode(
            ref="", role=AccessibilityRole.TOOLBAR, children=children
        )

        manager = RefManager()
        count = manager.assign_refs(root)

        assert count == 101
        assert root.ref == "@e1"
        assert children[0].ref == "@e2"
        assert children[99].ref == "@e101"


class TestRefManagerGetInteractiveNodes:
    """Test get_interactive_nodes method."""

    def test_get_interactive_nodes(self) -> None:
        """Returns all nodes with assigned refs."""
        child1 = AccessibilityNode(ref="", role=AccessibilityRole.BUTTON, name="A")
        child2 = AccessibilityNode(ref="", role=AccessibilityRole.LINK, name="B")
        root = AccessibilityNode(ref="", role=AccessibilityRole.DOCUMENT, children=[child1, child2])

        manager = RefManager()
        manager.assign_refs(root)

        nodes = manager.get_interactive_nodes()

        assert len(nodes) == 3
        names = {n.name for n in nodes}
        assert names == {None, "A", "B"}  # root has no name

    def test_get_interactive_nodes_with_filter(self) -> None:
        """Works correctly with interactive_only=True."""
        interactive = AccessibilityNode(
            ref="", role=AccessibilityRole.BUTTON, name="Click", is_interactive=True
        )
        static = AccessibilityNode(
            ref="", role=AccessibilityRole.STATIC_TEXT, name="Label", is_interactive=False
        )
        root = AccessibilityNode(
            ref="",
            role=AccessibilityRole.DOCUMENT,
            is_interactive=False,
            children=[interactive, static],
        )

        manager = RefManager()
        manager.assign_refs(root, interactive_only=True)

        nodes = manager.get_interactive_nodes()

        assert len(nodes) == 1
        assert nodes[0].name == "Click"
