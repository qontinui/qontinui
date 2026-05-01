"""Integration tests for UIA healing pipeline.

Verifies:
- VisionHealer tries UIA healing before visual search when accessibility_capture is provided
- UIA healing succeeds when automation_id matches a node in the tree
- UIA healing falls through to visual search when no match is found
- VisionHealer without accessibility_capture skips UIA entirely

All tests mock the IAccessibilityCapture layer so they run on ANY platform
(not just Windows) without needing a real UIA backend.
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
from qontinui_schemas.accessibility import AccessibilityBackend as AccBackend
from qontinui_schemas.accessibility import (
    AccessibilityBounds,
    AccessibilityNode,
    AccessibilityRole,
    AccessibilitySnapshot,
)

# Import directly to avoid circular import issues from healing/__init__.py
from qontinui.healing.healing_types import HealingContext, HealingStrategy
from qontinui.healing.vision_healer import VisionHealer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(
    ref: str,
    name: str,
    role: AccessibilityRole = AccessibilityRole.BUTTON,
    automation_id: str | None = None,
    bounds: AccessibilityBounds | None = None,
    is_interactive: bool = True,
) -> AccessibilityNode:
    """Create a minimal AccessibilityNode for testing."""
    return AccessibilityNode(
        ref=ref,
        role=role,
        name=name,
        automation_id=automation_id,
        bounds=bounds or AccessibilityBounds(x=100, y=200, width=80, height=30),
        is_interactive=is_interactive,
        children=[],
    )


def make_mock_capture(nodes: list[AccessibilityNode]) -> MagicMock:
    """Create a mock IAccessibilityCapture that returns the given nodes.

    ``is_connected`` is a regular method (not async) so we use MagicMock as
    the base and attach async methods explicitly.
    """
    capture = MagicMock()
    capture.is_connected.return_value = True

    root = AccessibilityNode(
        ref="@e0",
        role=AccessibilityRole.WINDOW,
        name="Test App",
        is_interactive=False,
        children=nodes,
    )
    snapshot = AccessibilitySnapshot(
        root=root,
        timestamp=1.0,
        backend=AccBackend.UIA,
        title="Test App",
        total_nodes=len(nodes) + 1,
        interactive_nodes=len(nodes),
    )
    capture.capture_tree = AsyncMock(return_value=snapshot)
    capture.find_nodes = AsyncMock(return_value=nodes)
    return capture


def _make_screenshot() -> np.ndarray:
    """Create a minimal black screenshot for tests that need one."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _make_context(**overrides) -> HealingContext:
    """Create a HealingContext with sensible defaults for UIA healing tests."""
    defaults = {
        "original_description": "Save button",
        "action_type": "click",
        "failure_reason": "Template not found",
        "additional_context": {
            "automation_id": "btnSave",
            "role": "button",
            "name": "Save",
        },
    }
    defaults.update(overrides)
    return HealingContext(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVisionHealerTriesUIABeforeVisualSearch:
    """VisionHealer with accessibility_capture attempts UIA first."""

    def test_vision_healer_tries_uia_before_visual_search(self):
        """Create VisionHealer with mock accessibility_capture.
        Call heal() and verify UIA healing is attempted before visual search."""
        nodes = [
            _make_node("@e1", "Save", automation_id="btnSave"),
        ]
        capture = make_mock_capture(nodes)

        healer = VisionHealer(accessibility_capture=capture)
        screenshot = _make_screenshot()
        context = _make_context()

        result = healer.heal(screenshot, context)

        # UIA healing should have succeeded (automation_id matches)
        assert result.success
        assert result.strategy == HealingStrategy.UIA_SELECTOR
        assert result.location is not None

        # Verify the location comes from the node's bounds (center of 100,200,80,30)
        assert result.location.x == 140  # 100 + 80//2
        assert result.location.y == 215  # 200 + 30//2


class TestVisionHealerUIAHealsByAutomationId:
    """UIA healing succeeds when automation_id matches a node."""

    def test_vision_healer_uia_heals_by_automation_id(self):
        """Mock tree where automation_id matches the context.
        Verify healing succeeds with UIA_SELECTOR strategy."""
        nodes = [
            _make_node("@e1", "Open", automation_id="btnOpen"),
            _make_node(
                "@e2",
                "Save",
                automation_id="btnSave",
                bounds=AccessibilityBounds(x=300, y=400, width=100, height=40),
            ),
            _make_node("@e3", "Close", automation_id="btnClose"),
        ]
        capture = make_mock_capture(nodes)

        healer = VisionHealer(accessibility_capture=capture)
        screenshot = _make_screenshot()
        context = _make_context()

        result = healer.heal(screenshot, context)

        assert result.success
        assert result.strategy == HealingStrategy.UIA_SELECTOR
        assert result.location is not None
        # Should find the Save button (node @e2) at center of (300, 400, 100, 40)
        assert result.location.x == 350  # 300 + 100//2
        assert result.location.y == 420  # 400 + 40//2
        assert "UIA" in (result.location.description or "")


class TestVisionHealerUIAFailsFallsThrough:
    """When UIA finds nothing, VisionHealer falls through to visual search."""

    def test_vision_healer_uia_fails_falls_through(self):
        """Mock tree with no matching elements. Verify it falls through
        to visual search (or fails if no pattern)."""
        # Nodes that do NOT match the context's automation_id="btnSave"
        nodes = [
            _make_node("@e1", "Open", automation_id="btnOpen"),
            _make_node("@e2", "Close", automation_id="btnClose"),
        ]
        capture = make_mock_capture(nodes)

        healer = VisionHealer(accessibility_capture=capture)
        screenshot = _make_screenshot()
        # Context looks for btnSave which doesn't exist in the tree.
        # Also use a name that won't fuzzy-match "Open" or "Close" at threshold 0.7
        context = _make_context(
            additional_context={
                "automation_id": "btnSave",
                "role": "button",
                "name": "Save",
            }
        )

        # No pattern, no LLM -> should fail after UIA fails
        result = healer.heal(screenshot, context, pattern=None)

        assert not result.success
        assert result.strategy == HealingStrategy.FAILED
        # Verify UIA was attempted (recorded in attempts list)
        uia_attempts = [
            (s, msg) for s, msg in result.attempts if s == HealingStrategy.UIA_SELECTOR
        ]
        assert len(uia_attempts) >= 1, "UIA strategy should have been attempted"


class TestVisionHealerWithoutAccessibilitySkipsUIA:
    """VisionHealer without accessibility_capture does not attempt UIA."""

    def test_vision_healer_without_accessibility_skips_uia(self):
        """Create VisionHealer without accessibility_capture.
        Verify UIA strategy is not attempted at all."""
        healer = VisionHealer()  # No accessibility_capture
        screenshot = _make_screenshot()
        context = _make_context()

        result = healer.heal(screenshot, context, pattern=None)

        # Should fail (no pattern, no LLM, no UIA)
        assert not result.success
        assert result.strategy == HealingStrategy.FAILED

        # No UIA attempt in the attempts list
        uia_attempts = [
            (s, msg) for s, msg in result.attempts if s == HealingStrategy.UIA_SELECTOR
        ]
        assert len(uia_attempts) == 0, "UIA strategy should NOT have been attempted"


class TestUIAHealerDirectly:
    """Test the UIAHealer class directly for more granular strategy coverage."""

    def test_uia_healer_automation_id_match(self):
        """UIAHealer finds element by automation_id."""
        from qontinui.healing.uia_healer import UIAHealer

        nodes = [
            _make_node("@e1", "Save", automation_id="btnSave"),
        ]
        capture = make_mock_capture(nodes)

        uia_healer = UIAHealer(capture)
        context = _make_context()

        result = uia_healer.heal(context)

        assert result.success
        assert result.strategy == HealingStrategy.UIA_SELECTOR
        assert "automation_id" in result.message

    def test_uia_healer_semantic_match(self):
        """UIAHealer falls back to semantic match (role + name)."""
        from qontinui.healing.uia_healer import UIAHealer

        # Node has same role and name but different automation_id
        nodes = [
            _make_node("@e1", "Save", automation_id="differentId"),
        ]
        capture = make_mock_capture(nodes)

        uia_healer = UIAHealer(capture)
        context = _make_context(
            additional_context={
                "automation_id": "btnSave",  # Won't match
                "role": "button",
                "name": "Save",  # Will match semantically
            }
        )

        result = uia_healer.heal(context)

        assert result.success
        assert result.strategy == HealingStrategy.UIA_SELECTOR
        assert "semantic" in result.message

    def test_uia_healer_all_strategies_fail(self):
        """UIAHealer returns failure when no strategy finds a match."""
        from qontinui.healing.uia_healer import UIAHealer

        # Node that won't match anything in the context
        nodes = [
            _make_node(
                "@e1",
                "Completely Different Element",
                role=AccessibilityRole.STATIC_TEXT,
                automation_id="txtInfo",
            ),
        ]
        capture = make_mock_capture(nodes)

        uia_healer = UIAHealer(capture)
        context = _make_context(
            additional_context={
                "automation_id": "btnSave",
                "role": "button",
                "name": "Save",
                "bounds": (900, 900, 80, 30),  # Far from (100, 200)
            }
        )

        result = uia_healer.heal(context)

        assert not result.success
        assert result.strategy == HealingStrategy.FAILED
        assert len(result.attempts) > 0  # Multiple strategies were tried

    def test_uia_healer_empty_tree(self):
        """UIAHealer handles empty accessibility tree gracefully."""
        from qontinui.healing.uia_healer import UIAHealer

        capture = make_mock_capture([])  # No interactive nodes

        uia_healer = UIAHealer(capture)
        context = _make_context()

        result = uia_healer.heal(context)

        assert not result.success
        assert result.strategy == HealingStrategy.FAILED

    def test_uia_healer_stats_tracking(self):
        """UIAHealer tracks statistics correctly."""
        from qontinui.healing.uia_healer import UIAHealer

        nodes = [_make_node("@e1", "Save", automation_id="btnSave")]
        capture = make_mock_capture(nodes)

        uia_healer = UIAHealer(capture)
        context = _make_context()

        # First heal: should succeed
        result1 = uia_healer.heal(context)
        assert result1.success

        stats = uia_healer.get_stats()
        assert stats["total"] == 1
        assert stats["success"] == 1
        assert stats["by_automation_id"] == 1
