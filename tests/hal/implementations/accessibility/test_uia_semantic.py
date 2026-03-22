"""Tests for UIA semantic search utilities."""

import sys
from pathlib import Path

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import time

from qontinui_schemas.accessibility import (
    AccessibilityBackend,
    AccessibilityNode,
    AccessibilityRole,
    AccessibilitySnapshot,
)

from qontinui.hal.implementations.accessibility.uia_semantic import (
    SemanticSearchCache,
    _extract_name_keywords,
    _extract_role_hints,
    format_nodes_for_llm,
    fuzzy_match_nodes,
)


def _make_node(
    ref: str = "@e1",
    role: AccessibilityRole = AccessibilityRole.BUTTON,
    name: str | None = None,
    value: str | None = None,
    automation_id: str | None = None,
    is_interactive: bool = True,
    children: list[AccessibilityNode] | None = None,
) -> AccessibilityNode:
    """Helper to build an AccessibilityNode for testing."""
    return AccessibilityNode(
        ref=ref,
        role=role,
        name=name,
        value=value,
        automation_id=automation_id,
        is_interactive=is_interactive,
        children=children or [],
    )


def _make_snapshot(
    nodes: list[AccessibilityNode],
) -> AccessibilitySnapshot:
    """Wrap a list of nodes under a root and return a snapshot."""
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


class TestExtractRoleHints:
    """Tests for _extract_role_hints."""

    def test_extract_role_hints_search_bar(self):
        """'the search bar' should include SEARCH and TEXTBOX roles."""
        roles = _extract_role_hints("the search bar")
        assert AccessibilityRole.SEARCH in roles
        assert AccessibilityRole.TEXTBOX in roles

    def test_extract_role_hints_button(self):
        """'button' should include BUTTON role."""
        roles = _extract_role_hints("click the submit button")
        assert AccessibilityRole.BUTTON in roles

    def test_extract_role_hints_no_match(self):
        """Description without role hints returns empty list."""
        roles = _extract_role_hints("some random text")
        assert roles == []


class TestExtractNameKeywords:
    """Tests for _extract_name_keywords."""

    def test_extract_name_keywords_basic(self):
        """'click the blue Submit button' should yield ['blue', 'submit']."""
        keywords = _extract_name_keywords("click the blue Submit button")
        assert "blue" in keywords
        assert "submit" in keywords
        # Filler words like 'click', 'the' should be stripped
        assert "click" not in keywords
        assert "the" not in keywords
        # Role hint 'button' should be stripped
        assert "button" not in keywords

    def test_extract_name_keywords_empty(self):
        """Pure filler description yields empty list."""
        keywords = _extract_name_keywords("the a an")
        assert keywords == []


class TestFuzzyMatchNodes:
    """Tests for fuzzy_match_nodes."""

    def test_fuzzy_match_exact_name(self):
        """Node with name='Save' matches description='Save' with score 1.0."""
        node = _make_node(ref="@e1", name="Save")
        snapshot = _make_snapshot([node])

        matches = fuzzy_match_nodes("Save", snapshot)
        assert len(matches) >= 1
        assert matches[0].node.ref == "@e1"
        assert matches[0].score == 1.0

    def test_fuzzy_match_keyword_match(self):
        """Node with name='Save Document' matches 'Save' with high score."""
        node = _make_node(ref="@e1", name="Save Document")
        snapshot = _make_snapshot([node])

        matches = fuzzy_match_nodes("Save", snapshot, min_score=0.3)
        assert len(matches) >= 1
        assert matches[0].node.ref == "@e1"
        assert matches[0].score >= 0.5

    def test_fuzzy_match_role_hint_bonus(self):
        """Button node gets role bonus when description mentions 'button'."""
        btn = _make_node(ref="@e1", role=AccessibilityRole.BUTTON, name="OK")
        link = _make_node(ref="@e2", role=AccessibilityRole.LINK, name="OK")
        snapshot = _make_snapshot([btn, link])

        matches = fuzzy_match_nodes("OK button", snapshot, min_score=0.3)
        # Both match on name, but the button should score higher via role bonus
        btn_match = next(m for m in matches if m.node.ref == "@e1")
        link_match = next((m for m in matches if m.node.ref == "@e2"), None)
        if link_match is not None:
            assert btn_match.score > link_match.score

    def test_fuzzy_match_automation_id(self):
        """Matches against automation_id."""
        node = _make_node(
            ref="@e1",
            name="",
            automation_id="btnSubmitForm",
        )
        snapshot = _make_snapshot([node])

        matches = fuzzy_match_nodes("submit", snapshot, min_score=0.3)
        assert len(matches) >= 1
        assert matches[0].match_type == "automation_id"

    def test_fuzzy_match_min_score_filter(self):
        """Low-similarity nodes are filtered out by min_score."""
        node = _make_node(ref="@e1", name="Completely Unrelated Name")
        snapshot = _make_snapshot([node])

        matches = fuzzy_match_nodes("xyzzy", snapshot, min_score=0.8)
        assert len(matches) == 0


class TestFormatNodesForLLM:
    """Tests for format_nodes_for_llm."""

    def test_format_nodes_for_llm(self):
        """Outputs indexed list format with refs, roles, and names."""
        nodes = [
            _make_node(ref="@e1", role=AccessibilityRole.BUTTON, name="Save"),
            _make_node(
                ref="@e2",
                role=AccessibilityRole.TEXTBOX,
                name="Search",
                value="",
                automation_id="searchBox",
            ),
        ]
        snapshot = _make_snapshot(nodes)

        output = format_nodes_for_llm(snapshot)
        lines = output.strip().split("\n")
        assert len(lines) == 2
        assert "[0]" in lines[0]
        assert "@e1" in lines[0]
        assert "button" in lines[0].lower()
        assert '"Save"' in lines[0]
        assert "[1]" in lines[1]
        assert "@e2" in lines[1]
        assert "[id=searchBox]" in lines[1]


class TestSemanticSearchCache:
    """Tests for SemanticSearchCache put/get/invalidate."""

    def test_put_get_roundtrip(self):
        """Put results, then get them back."""
        cache = SemanticSearchCache()
        node = _make_node(ref="@e1", name="Save")
        from qontinui.hal.implementations.accessibility.uia_semantic import SemanticMatch

        results = [
            SemanticMatch(node=node, score=0.9, match_type="exact_name", matched_term="Save")
        ]
        cache.put("notepad", "Save button", results)

        fetched = cache.get("notepad", "Save button")
        assert fetched is not None
        assert len(fetched) == 1
        assert fetched[0].score == 0.9

    def test_get_miss(self):
        """Get on an empty cache returns None."""
        cache = SemanticSearchCache()
        assert cache.get("notepad", "Save button") is None

    def test_invalidate_by_app(self):
        """Invalidate by app_key removes only that app's entries."""
        cache = SemanticSearchCache()
        cache.put("notepad", "Save", [])
        cache.put("chrome", "Back", [])

        cache.invalidate("notepad")
        assert cache.get("notepad", "Save") is None
        assert cache.get("chrome", "Back") is not None

    def test_invalidate_all(self):
        """Invalidate without app_key clears everything."""
        cache = SemanticSearchCache()
        cache.put("notepad", "Save", [])
        cache.put("chrome", "Back", [])

        cache.invalidate()
        assert cache.get("notepad", "Save") is None
        assert cache.get("chrome", "Back") is None
