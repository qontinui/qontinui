"""Tests for RefManager persistence (save/load) and registration."""

import json
import sys
from pathlib import Path

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from qontinui_schemas.accessibility import AccessibilityBounds, AccessibilityNode, AccessibilityRole

from qontinui.hal.implementations.accessibility.ref_manager import RefManager


def _make_node(
    ref: str = "@e1",
    role: AccessibilityRole = AccessibilityRole.BUTTON,
    name: str | None = None,
    automation_id: str | None = None,
    class_name: str | None = None,
    bounds: AccessibilityBounds | None = None,
    is_interactive: bool = True,
    children: list[AccessibilityNode] | None = None,
) -> AccessibilityNode:
    """Helper to build an AccessibilityNode."""
    return AccessibilityNode(
        ref=ref,
        role=role,
        name=name,
        automation_id=automation_id,
        class_name=class_name,
        bounds=bounds,
        is_interactive=is_interactive,
        children=children or [],
    )


class TestRefManagerPersistence:
    """Tests for RefManager save() and load()."""

    def test_save_load_roundtrip(self, tmp_path: Path):
        """Save refs, load them, verify fingerprints match."""
        manager = RefManager()

        node1 = _make_node(
            ref="@e1",
            role=AccessibilityRole.BUTTON,
            name="Save",
            automation_id="btnSave",
            class_name="Button",
            bounds=AccessibilityBounds(x=10, y=20, width=80, height=30),
        )
        node2 = _make_node(
            ref="@e2",
            role=AccessibilityRole.TEXTBOX,
            name="Search",
            automation_id="txtSearch",
        )

        manager.register_node("@e1", node1)
        manager.register_node("@e2", node2)

        save_path = tmp_path / "refs.json"
        count = manager.save(save_path)
        assert count == 2

        loaded = manager.load(save_path)
        assert "@e1" in loaded
        assert "@e2" in loaded
        assert loaded["@e1"]["automation_id"] == "btnSave"
        assert loaded["@e1"]["role"] == "button"
        assert loaded["@e1"]["name"] == "Save"
        assert loaded["@e1"]["bounds"] == [10, 20, 80, 30]
        assert loaded["@e2"]["automation_id"] == "txtSearch"
        assert loaded["@e2"]["role"] == "textbox"

    def test_load_missing_file(self, tmp_path: Path):
        """Loading from a missing file returns empty dict."""
        manager = RefManager()
        result = manager.load(tmp_path / "nonexistent.json")
        assert result == {}

    def test_load_corrupt_json(self, tmp_path: Path):
        """Loading corrupt JSON returns empty dict without crashing."""
        manager = RefManager()
        bad_file = tmp_path / "corrupt.json"
        bad_file.write_text("{invalid json content!!!", encoding="utf-8")

        result = manager.load(bad_file)
        assert result == {}

    def test_register_node(self):
        """Manually register a node, then find by ref."""
        manager = RefManager()
        node = _make_node(ref="@tmp", name="OK")

        manager.register_node("@e5", node)

        found = manager.get_node_by_ref("@e5")
        assert found is not None
        assert found.name == "OK"
        assert found.ref == "@e5"

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        """Parent directories are created automatically on save."""
        manager = RefManager()
        node = _make_node(ref="@e1", name="Test")
        manager.register_node("@e1", node)

        nested_path = tmp_path / "deep" / "nested" / "dir" / "refs.json"
        count = manager.save(nested_path)

        assert count == 1
        assert nested_path.exists()

        data = json.loads(nested_path.read_text(encoding="utf-8"))
        assert "@e1" in data
