"""Integration tests for cascade detection with accessibility backends.

Verifies:
- CascadeDetector includes accessibility backends when accessibility_capture is provided
- AccessibilityBackend is tried first (lowest cost at ~5ms)
- Fallback to template matching when accessibility returns no results
- SemanticAccessibilityBackend handles "description" needle type
- Backend ordering: accessibility (5ms) < semantic (10ms) < template (~20ms)
- CascadeDetector without accessibility_capture skips accessibility backends

All tests mock the IAccessibilityCapture layer so they run on ANY platform
(not just Windows) without needing a real UIA/CDP backend.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

from qontinui_schemas.accessibility import AccessibilityBackend as AccBackend
from qontinui_schemas.accessibility import (
    AccessibilityBounds,
    AccessibilityNode,
    AccessibilityRole,
    AccessibilitySnapshot,
)

from qontinui.find.backends.base import DetectionBackend, DetectionResult
from qontinui.find.backends.cascade import CascadeDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_capture(nodes: list[AccessibilityNode]) -> MagicMock:
    """Create a mock IAccessibilityCapture that returns the given nodes.

    The mock exposes ``is_connected()`` (sync), ``capture_tree()`` (async),
    and ``find_nodes()`` (async) — the three methods used by accessibility
    backends.

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
    # Async methods
    capture.capture_tree = AsyncMock(return_value=snapshot)
    capture.find_nodes = AsyncMock(return_value=nodes)
    return capture


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


class StubBackend(DetectionBackend):
    """Minimal stub backend for verifying cascade behaviour."""

    def __init__(
        self,
        backend_name: str,
        cost: float,
        supported_types: list[str],
        results: list[DetectionResult] | None = None,
        available: bool = True,
    ):
        self._name = backend_name
        self._cost = cost
        self._supported = supported_types
        self._results = results or []
        self._available = available
        self.find_called = False
        self.find_call_count = 0

    def find(self, needle, haystack, config):
        self.find_called = True
        self.find_call_count += 1
        return list(self._results)

    def supports(self, needle_type):
        return needle_type in self._supported

    def estimated_cost_ms(self):
        return self._cost

    @property
    def name(self):
        return self._name

    def is_available(self):
        return self._available


def _make_result(confidence: float = 0.9, backend: str = "test") -> DetectionResult:
    return DetectionResult(
        x=100, y=200, width=50, height=30, confidence=confidence, backend_name=backend
    )


# ---------------------------------------------------------------------------
# Tests: Cascade with accessibility backends
# ---------------------------------------------------------------------------


class TestCascadeWithAccessibilityTriesAccessibilityFirst:
    """CascadeDetector with accessibility_capture puts AccessibilityBackend first."""

    def test_cascade_with_accessibility_tries_accessibility_first(self):
        """When accessibility_capture is provided and needle_type is 'label',
        the AccessibilityBackend should be tried first and return results."""
        nodes = [
            _make_node("@e1", "Save", automation_id="btnSave"),
            _make_node("@e2", "Cancel", automation_id="btnCancel"),
        ]
        capture = make_mock_capture(nodes)

        cascade = CascadeDetector(accessibility_capture=capture)

        # Verify accessibility backend is present
        backend_names = [b.name for b in cascade.backends]
        assert "accessibility" in backend_names

        # The accessibility backend should be first (cost=5ms)
        assert cascade.backends[0].name == "accessibility"

        # Search by label — should hit accessibility backend
        results = cascade.find("Save", None, {"needle_type": "label"})

        assert len(results) >= 1
        assert results[0].backend_name == "accessibility"
        assert results[0].confidence == 1.0


class TestCascadeWithAccessibilityFallsBackToTemplate:
    """When accessibility returns nothing, cascade falls through to template."""

    def test_cascade_with_accessibility_falls_back_to_template(self):
        """Mock accessibility_capture that returns no results for find_nodes.
        Verify template matching is tried next."""
        # Empty node list -> accessibility will find nothing
        capture = make_mock_capture([])

        # Build cascade with explicit backends: accessibility (empty) + stub template
        from qontinui.find.backends.accessibility_backend import AccessibilityBackend

        acc_backend = AccessibilityBackend(capture)
        template_stub = StubBackend(
            "template_match",
            cost=20,
            supported_types=["label", "template"],
            results=[_make_result(0.95, "template_match")],
        )

        cascade = CascadeDetector(backends=[acc_backend, template_stub])
        results = cascade.find("Save", None, {"needle_type": "label"})

        # Accessibility returned nothing, template should have been tried
        assert template_stub.find_called
        assert len(results) == 1
        assert results[0].backend_name == "template_match"


class TestCascadeWithoutAccessibilitySkipsIt:
    """CascadeDetector without accessibility_capture has no accessibility backends."""

    def test_cascade_without_accessibility_skips_it(self):
        """Create CascadeDetector without accessibility_capture.
        Verify only template/feature backends are present (no accessibility)."""
        cascade = CascadeDetector()

        backend_names = [b.name for b in cascade.backends]
        assert "accessibility" not in backend_names
        assert "semantic_accessibility" not in backend_names

        # Should still have template-based backends
        assert len(cascade.backends) > 0


class TestCascadeSemanticAccessibilityFindsByDescription:
    """SemanticAccessibilityBackend finds elements by natural-language description."""

    def test_cascade_semantic_accessibility_finds_by_description(self):
        """Search by 'description' needle_type with a natural-language query.
        Verify SemanticAccessibilityBackend handles it."""
        nodes = [
            _make_node("@e1", "Save Document", automation_id="btnSave"),
            _make_node("@e2", "Cancel", automation_id="btnCancel"),
        ]
        capture = make_mock_capture(nodes)

        # We mock the fuzzy_match_nodes import inside SemanticAccessibilityBackend
        # to return a match for our description.
        mock_match = MagicMock()
        mock_match.score = 0.85
        mock_match.node = nodes[0]
        mock_match.match_type = "name"
        mock_match.matched_term = "Save Document"

        with patch(
            "qontinui.find.backends.semantic_accessibility_backend.SemanticAccessibilityBackend._get_cache"
        ) as mock_get_cache:
            # Cache returns None (no hit)
            mock_cache = MagicMock()
            mock_cache.get.return_value = None
            mock_get_cache.return_value = mock_cache

            with patch(
                "qontinui.hal.implementations.accessibility.uia_semantic.fuzzy_match_nodes",
                return_value=[mock_match],
            ):
                from qontinui.find.backends.semantic_accessibility_backend import (
                    SemanticAccessibilityBackend,
                )

                sem_backend = SemanticAccessibilityBackend(capture)
                results = sem_backend.find(
                    "the save button",
                    None,
                    {"needle_type": "description", "min_confidence": 0.6},
                )

                assert len(results) >= 1
                assert results[0].backend_name == "semantic_accessibility"
                assert results[0].confidence == 0.85
                assert results[0].label == "Save Document"


class TestCascadeBackendOrdering:
    """Backends are sorted by cost: accessibility < semantic < template."""

    def test_cascade_backend_ordering(self):
        """Verify backends are sorted by estimated cost when accessibility_capture
        is provided. Accessibility (5ms) < semantic (10ms) < template (~20ms)."""
        nodes = [_make_node("@e1", "OK")]
        capture = make_mock_capture(nodes)

        cascade = CascadeDetector(accessibility_capture=capture)

        costs = [b.estimated_cost_ms() for b in cascade.backends]

        # Verify costs are in ascending order
        assert costs == sorted(costs), f"Backends not sorted by cost: {costs}"

        # Verify accessibility is cheapest
        assert cascade.backends[0].name == "accessibility"
        assert cascade.backends[0].estimated_cost_ms() == 5.0

        # Find semantic backend and verify its cost
        sem_backends = [b for b in cascade.backends if b.name == "semantic_accessibility"]
        assert len(sem_backends) == 1
        assert sem_backends[0].estimated_cost_ms() == 10.0

        # Semantic should come before template-based backends
        sem_idx = next(
            i for i, b in enumerate(cascade.backends) if b.name == "semantic_accessibility"
        )
        template_indices = [i for i, b in enumerate(cascade.backends) if "template" in b.name]
        if template_indices:
            assert sem_idx < min(
                template_indices
            ), "Semantic backend should come before template backends"


class TestCascadeTimeBudget:
    def test_max_time_ms_stops_cascade(self):
        """When max_time_ms is set, cascade stops after time budget is exhausted."""

        # Create a slow backend that sleeps
        class SlowBackend(StubBackend):
            def find(self, needle, haystack, config):
                self.find_called = True
                self.find_call_count += 1
                time.sleep(0.1)  # 100ms
                return []

        slow1 = SlowBackend("slow1", cost=10, supported_types=["template"])
        slow2 = SlowBackend("slow2", cost=20, supported_types=["template"])
        fast_result = StubBackend(
            "fast_result",
            cost=30,
            supported_types=["template"],
            results=[_make_result(0.95, "fast_result")],
        )

        cascade = CascadeDetector(backends=[slow1, slow2, fast_result])
        results = cascade.find(
            "x",
            None,
            {
                "needle_type": "template",
                "max_time_ms": 50,  # Only 50ms budget — should stop after slow1
            },
        )

        # slow1 should have run (it takes 100ms but we check BEFORE running)
        assert slow1.find_called
        # fast_result should NOT have been reached due to time budget
        assert not fast_result.find_called
        assert len(results) == 0

    def test_max_time_ms_none_runs_all(self):
        """When max_time_ms is None, all backends are tried."""
        b1 = StubBackend("b1", cost=10, supported_types=["template"])
        b2 = StubBackend(
            "b2", cost=20, supported_types=["template"], results=[_make_result(0.95, "b2")]
        )

        cascade = CascadeDetector(backends=[b1, b2])
        results = cascade.find("x", None, {"needle_type": "template"})

        assert b1.find_called
        assert b2.find_called
        assert len(results) == 1
