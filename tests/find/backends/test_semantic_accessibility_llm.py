"""Tests for the LLM fallback path in SemanticAccessibilityBackend.

Verifies:
- LLM fallback is triggered when fuzzy match score < llm_threshold
- LLM fallback is NOT triggered when fuzzy score >= llm_threshold
- LLM fallback is NOT triggered when no llm_client is provided
- _llm_find correctly parses valid LLM responses (INDEX + CONFIDENCE)
- _llm_find handles edge cases: none index, out-of-bounds, low confidence,
  malformed response
- estimated_cost_ms reflects whether llm_client is present
"""

from unittest.mock import AsyncMock, MagicMock, patch

from qontinui_schemas.accessibility import AccessibilityBackend as AccBackend
from qontinui_schemas.accessibility import (
    AccessibilityBounds,
    AccessibilityNode,
    AccessibilityRole,
    AccessibilitySnapshot,
)

from qontinui.find.backends.semantic_accessibility_backend import SemanticAccessibilityBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NODES = [
    AccessibilityNode(
        ref="@e1",
        role=AccessibilityRole.BUTTON,
        name="Cancel",
        bounds=AccessibilityBounds(x=10, y=10, width=80, height=30),
        is_interactive=True,
        children=[],
    ),
    AccessibilityNode(
        ref="@e2",
        role=AccessibilityRole.TEXTBOX,
        name="Username",
        bounds=AccessibilityBounds(x=10, y=50, width=200, height=30),
        is_interactive=True,
        children=[],
    ),
    AccessibilityNode(
        ref="@e3",
        role=AccessibilityRole.BUTTON,
        name="Submit",
        bounds=AccessibilityBounds(x=10, y=90, width=80, height=30),
        is_interactive=True,
        children=[],
    ),
]


def _make_capture(nodes: list[AccessibilityNode] | None = None) -> MagicMock:
    """Create a mock IAccessibilityCapture returning the given nodes."""
    if nodes is None:
        nodes = list(NODES)

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
    return capture


def _make_llm_client(response: str = "INDEX: 0\nCONFIDENCE: 0.9") -> AsyncMock:
    """Create a mock LLM client."""
    client = AsyncMock()
    client.complete = AsyncMock(return_value=response)
    return client


def _low_score_match(node: AccessibilityNode, score: float = 0.3) -> MagicMock:
    """Create a mock SemanticMatch with a low score."""
    m = MagicMock()
    m.score = score
    m.node = node
    m.match_type = "name"
    m.matched_term = node.name
    return m


def _high_score_match(node: AccessibilityNode, score: float = 0.9) -> MagicMock:
    """Create a mock SemanticMatch with a high score."""
    m = MagicMock()
    m.score = score
    m.node = node
    m.match_type = "name"
    m.matched_term = node.name
    return m


def _patch_fuzzy(return_value):
    """Patch fuzzy_match_nodes inside the semantic backend module."""
    return patch(
        "qontinui.hal.implementations.accessibility.uia_semantic.fuzzy_match_nodes",
        return_value=return_value,
    )


def _patch_cache():
    """Patch _get_cache so no real SemanticSearchCache is needed."""
    mock_cache = MagicMock()
    mock_cache.get.return_value = None  # no cache hit
    return patch.object(
        SemanticAccessibilityBackend,
        "_get_cache",
        return_value=mock_cache,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLlmFallbackTriggeredWhenFuzzyBelowThreshold:
    def test_llm_fallback_triggered_when_fuzzy_below_threshold(self):
        """When best fuzzy score < llm_threshold and llm_client present,
        the LLM client's complete() should be invoked."""
        capture = _make_capture()
        llm_client = _make_llm_client("INDEX: 2\nCONFIDENCE: 0.95")
        backend = SemanticAccessibilityBackend(capture, llm_client=llm_client, llm_threshold=0.7)

        low_matches = [_low_score_match(NODES[0], 0.3)]

        with _patch_cache(), _patch_fuzzy(low_matches):
            results = backend.find(
                "the submit button",
                None,
                {"needle_type": "description", "min_confidence": 0.5},
            )

        # LLM should have been called
        llm_client.complete.assert_awaited_once()
        # Should return a result from the LLM path
        assert len(results) == 1
        assert results[0].metadata["match_type"] == "llm"


class TestLlmFallbackNotTriggeredWhenFuzzyAboveThreshold:
    def test_llm_fallback_not_triggered_when_fuzzy_above_threshold(self):
        """When best fuzzy score >= llm_threshold, LLM is NOT called."""
        capture = _make_capture()
        llm_client = _make_llm_client()
        backend = SemanticAccessibilityBackend(capture, llm_client=llm_client, llm_threshold=0.7)

        high_matches = [_high_score_match(NODES[2], 0.9)]

        with _patch_cache(), _patch_fuzzy(high_matches):
            results = backend.find(
                "Submit",
                None,
                {"needle_type": "description", "min_confidence": 0.5},
            )

        llm_client.complete.assert_not_awaited()
        assert len(results) == 1
        assert results[0].confidence == 0.9


class TestLlmFallbackNotTriggeredWithoutClient:
    def test_llm_fallback_not_triggered_without_client(self):
        """Without llm_client, the LLM path is never taken even with low scores."""
        capture = _make_capture()
        backend = SemanticAccessibilityBackend(capture, llm_client=None)

        low_matches = [_low_score_match(NODES[0], 0.3)]

        with _patch_cache(), _patch_fuzzy(low_matches):
            results = backend.find(
                "something obscure",
                None,
                {"needle_type": "description", "min_confidence": 0.1},
            )

        # Should return the fuzzy result, not an LLM result
        assert len(results) == 1
        assert results[0].metadata.get("match_type") == "name"


class TestLlmFindParsesValidResponse:
    def test_llm_find_parses_valid_response(self):
        """LLM returning 'INDEX: 2\\nCONFIDENCE: 0.95' should yield the node
        at index 2 with confidence 0.95 and match_type 'llm'."""
        capture = _make_capture()
        llm_client = _make_llm_client("INDEX: 2\nCONFIDENCE: 0.95")
        backend = SemanticAccessibilityBackend(capture, llm_client=llm_client, llm_threshold=0.7)

        low_matches = [_low_score_match(NODES[0], 0.2)]

        with _patch_cache(), _patch_fuzzy(low_matches):
            results = backend.find(
                "the submit button",
                None,
                {"needle_type": "description", "min_confidence": 0.5},
            )

        assert len(results) == 1
        result = results[0]
        assert result.confidence == 0.95
        assert result.label == "Submit"
        assert result.metadata["match_type"] == "llm"
        assert result.metadata["ref"] == "@e3"
        assert result.x == 10
        assert result.y == 90
        assert result.width == 80
        assert result.height == 30


class TestLlmFindHandlesNoneResponse:
    def test_llm_find_handles_none_response(self):
        """LLM returning 'INDEX: none' should yield empty results."""
        capture = _make_capture()
        llm_client = _make_llm_client("INDEX: none\nCONFIDENCE: 0.0")
        backend = SemanticAccessibilityBackend(capture, llm_client=llm_client, llm_threshold=0.7)

        low_matches = [_low_score_match(NODES[0], 0.2)]

        with _patch_cache(), _patch_fuzzy(low_matches):
            results = backend.find(
                "nonexistent element",
                None,
                {"needle_type": "description", "min_confidence": 0.5},
            )

        llm_client.complete.assert_awaited_once()
        # LLM said "none", so no LLM results; falls back to fuzzy (score 0.2 < 0.5)
        assert len(results) == 0


class TestLlmFindHandlesOutOfBoundsIndex:
    def test_llm_find_handles_out_of_bounds_index(self):
        """LLM returning an index beyond the node list should yield empty results."""
        capture = _make_capture()
        llm_client = _make_llm_client("INDEX: 999\nCONFIDENCE: 0.8")
        backend = SemanticAccessibilityBackend(capture, llm_client=llm_client, llm_threshold=0.7)

        low_matches = [_low_score_match(NODES[0], 0.2)]

        with _patch_cache(), _patch_fuzzy(low_matches):
            results = backend.find(
                "phantom button",
                None,
                {"needle_type": "description", "min_confidence": 0.5},
            )

        llm_client.complete.assert_awaited_once()
        assert len(results) == 0


class TestLlmFindHandlesLowConfidence:
    def test_llm_find_handles_low_confidence(self):
        """LLM returning confidence below min_confidence should yield empty results."""
        capture = _make_capture()
        llm_client = _make_llm_client("INDEX: 1\nCONFIDENCE: 0.3")
        backend = SemanticAccessibilityBackend(capture, llm_client=llm_client, llm_threshold=0.7)

        low_matches = [_low_score_match(NODES[0], 0.2)]

        with _patch_cache(), _patch_fuzzy(low_matches):
            results = backend.find(
                "username field",
                None,
                {"needle_type": "description", "min_confidence": 0.5},
            )

        llm_client.complete.assert_awaited_once()
        # LLM confidence 0.3 < min_confidence 0.5 => no results
        assert len(results) == 0


class TestLlmFindHandlesMalformedResponse:
    def test_llm_find_handles_malformed_response(self):
        """LLM returning garbage text should yield empty results without crash."""
        capture = _make_capture()
        llm_client = _make_llm_client("Sorry, I cannot help with that request.")
        backend = SemanticAccessibilityBackend(capture, llm_client=llm_client, llm_threshold=0.7)

        low_matches = [_low_score_match(NODES[0], 0.2)]

        with _patch_cache(), _patch_fuzzy(low_matches):
            results = backend.find(
                "something",
                None,
                {"needle_type": "description", "min_confidence": 0.5},
            )

        llm_client.complete.assert_awaited_once()
        assert len(results) == 0


class TestEstimatedCostWithLlmClient:
    def test_estimated_cost_with_llm_client(self):
        """Backend with llm_client reports 200ms; without reports 10ms."""
        capture = _make_capture()

        with_llm = SemanticAccessibilityBackend(capture, llm_client=_make_llm_client())
        without_llm = SemanticAccessibilityBackend(capture, llm_client=None)

        assert with_llm.estimated_cost_ms() == 200.0
        assert without_llm.estimated_cost_ms() == 10.0
