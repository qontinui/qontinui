"""Tests for UIA-specific self-healing healer."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import time

from qontinui_schemas.accessibility import (
    AccessibilityBackend,
    AccessibilityBounds,
    AccessibilityNode,
    AccessibilityRole,
    AccessibilitySnapshot,
)

from qontinui.healing.healing_types import (
    HealingContext,
    HealingStrategy,
)
from qontinui.healing.uia_healer import (
    UIAElementFingerprint,
    UIAHealer,
)


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


def _make_snapshot(
    nodes: list[AccessibilityNode],
) -> AccessibilitySnapshot:
    """Wrap nodes under a root and return a snapshot."""
    root = _make_node(
        ref="@root",
        role=AccessibilityRole.APPLICATION,
        name="Test App",
        is_interactive=False,
        children=nodes,
    )
    return AccessibilitySnapshot(
        root=root,
        timestamp=time.time(),
        backend=AccessibilityBackend.UIA,
        total_nodes=1 + len(nodes),
        interactive_nodes=sum(1 for n in nodes if n.is_interactive),
    )


def _make_mock_capture(snapshot: AccessibilitySnapshot) -> MagicMock:
    """Create an AsyncMock IAccessibilityCapture that returns the given snapshot."""
    capture = MagicMock()
    capture.capture_tree = AsyncMock(return_value=snapshot)
    return capture


class TestUIAHealerStrategies:
    """Tests for UIAHealer healing strategies."""

    def test_heal_by_automation_id(self):
        """Finds element by matching automation_id."""
        target = _make_node(
            ref="@e1",
            role=AccessibilityRole.BUTTON,
            name="Submit",
            automation_id="btnSubmit",
            bounds=AccessibilityBounds(x=100, y=200, width=80, height=30),
        )
        snapshot = _make_snapshot([target])
        capture = _make_mock_capture(snapshot)
        healer = UIAHealer(capture)

        context = HealingContext(
            original_description="Submit",
            additional_context={
                "automation_id": "btnSubmit",
                "role": "button",
                "name": "Submit",
            },
        )

        result = healer.heal(context)
        assert result.success
        assert result.strategy == HealingStrategy.UIA_SELECTOR
        assert "by_automation_id" in result.message
        assert result.location is not None

    def test_heal_by_semantic_match(self):
        """Finds element by role + name when automation_id fails."""
        target = _make_node(
            ref="@e1",
            role=AccessibilityRole.BUTTON,
            name="Save Document",
            automation_id="newAutomationId",
            bounds=AccessibilityBounds(x=100, y=200, width=80, height=30),
        )
        snapshot = _make_snapshot([target])
        capture = _make_mock_capture(snapshot)
        healer = UIAHealer(capture)

        context = HealingContext(
            original_description="Save Document",
            additional_context={
                "automation_id": "oldMissingId",
                "role": "button",
                "name": "Save Document",
            },
        )

        result = healer.heal(context)
        assert result.success
        assert result.strategy == HealingStrategy.UIA_SELECTOR
        assert "by_semantic" in result.message

    def test_heal_by_structural_match(self):
        """Finds element by role + position zone when semantic fails."""
        target = _make_node(
            ref="@e1",
            role=AccessibilityRole.BUTTON,
            name="Totally Different Name",
            bounds=AccessibilityBounds(x=500, y=400, width=80, height=30),
        )
        snapshot = _make_snapshot([target])
        capture = _make_mock_capture(snapshot)
        healer = UIAHealer(capture)

        context = HealingContext(
            original_description="Original Name",
            additional_context={
                "automation_id": "missingId",
                "role": "button",
                "name": "Original Name",
                "bounds": (500, 400, 80, 30),
            },
        )

        result = healer.heal(context)
        assert result.success
        assert result.strategy == HealingStrategy.UIA_SELECTOR
        assert "by_structural" in result.message

    def test_heal_by_fuzzy_name(self):
        """Finds element by fuzzy name similarity."""
        target = _make_node(
            ref="@e1",
            role=AccessibilityRole.BUTTON,
            name="Save Changes",
            bounds=AccessibilityBounds(x=100, y=200, width=80, height=30),
        )
        # Add a different-role decoy in a different zone so structural doesn't match
        decoy = _make_node(
            ref="@e2",
            role=AccessibilityRole.LINK,
            name="Other",
            bounds=AccessibilityBounds(x=100, y=900, width=80, height=30),
        )
        snapshot = _make_snapshot([target, decoy])
        capture = _make_mock_capture(snapshot)
        healer = UIAHealer(capture, fuzzy_threshold=0.5)

        context = HealingContext(
            original_description="Save Changez",
            additional_context={
                "automation_id": "missingId",
                "role": "button",
                "name": "Save Changez",
                # Put bounds in a different zone so structural doesn't match
                "bounds": (100, 900, 80, 30),
            },
        )

        result = healer.heal(context)
        assert result.success
        assert result.strategy == HealingStrategy.UIA_SELECTOR
        assert "by_fuzzy" in result.message

    def test_heal_all_fail(self):
        """Returns FAILED when no strategy matches."""
        node = _make_node(
            ref="@e1",
            role=AccessibilityRole.LINK,
            name="Unrelated",
            bounds=AccessibilityBounds(x=100, y=200, width=80, height=30),
        )
        snapshot = _make_snapshot([node])
        capture = _make_mock_capture(snapshot)
        healer = UIAHealer(capture, fuzzy_threshold=0.99)

        context = HealingContext(
            original_description="NonexistentElement",
            additional_context={
                "automation_id": "doesNotExist",
                "role": "button",
                "name": "XYZZY",
                "bounds": (9999, 9999, 10, 10),
            },
        )

        result = healer.heal(context)
        assert not result.success
        assert result.strategy == HealingStrategy.FAILED
        assert len(result.attempts) > 0


class TestUIAElementFingerprint:
    """Tests for UIAElementFingerprint."""

    def test_fingerprint_from_node(self):
        """from_node() extracts fields correctly."""
        node = _make_node(
            ref="@e1",
            role=AccessibilityRole.TEXTBOX,
            name="Email",
            automation_id="txtEmail",
            class_name="TextInput",
            bounds=AccessibilityBounds(x=10, y=20, width=200, height=30),
        )

        fp = UIAElementFingerprint.from_node(node)
        assert fp.automation_id == "txtEmail"
        assert fp.role == "textbox"
        assert fp.name == "Email"
        assert fp.class_name == "TextInput"
        assert fp.bounds == (10, 20, 200, 30)

    def test_fingerprint_from_context(self):
        """from_context() extracts fields correctly."""
        context = HealingContext(
            original_description="Email input",
            additional_context={
                "automation_id": "txtEmail",
                "role": "textbox",
                "name": "Email",
                "class_name": "TextInput",
                "bounds": (10, 20, 200, 30),
            },
        )

        fp = UIAElementFingerprint.from_context(context)
        assert fp.automation_id == "txtEmail"
        assert fp.role == "textbox"
        assert fp.name == "Email"
        assert fp.class_name == "TextInput"
        assert fp.bounds == (10, 20, 200, 30)

    def test_fingerprint_from_context_fallback_name(self):
        """from_context() falls back to original_description when name is missing."""
        context = HealingContext(
            original_description="Email input",
            additional_context={"role": "textbox"},
        )

        fp = UIAElementFingerprint.from_context(context)
        assert fp.name == "Email input"
