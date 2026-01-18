"""
Tests for accessibility_extractor module.

Tests accessibility tree extraction and merging with DOM data.
"""

import pytest

from qontinui.extraction.web.accessibility_extractor import (
    A11yNode,
    A11yTree,
    AccessibilityExtractor,
    EnrichedElement,
    a11y_tree_to_text,
)
from qontinui.extraction.web.models import BoundingBox, InteractiveElement


class TestA11yNode:
    """Tests for A11yNode dataclass."""

    def test_create_node(self) -> None:
        """Test creating an A11yNode."""
        node = A11yNode(
            role="button",
            name="Submit",
            description="Submit the form",
        )

        assert node.role == "button"
        assert node.name == "Submit"
        assert node.description == "Submit the form"

    def test_create_node_with_state(self) -> None:
        """Test creating a node with state properties."""
        node = A11yNode(
            role="checkbox",
            name="Accept terms",
            description="",
            checked=True,
            disabled=False,
        )

        assert node.checked is True
        assert node.disabled is False

    def test_from_playwright(self) -> None:
        """Test creating from Playwright snapshot data."""
        pw_data = {
            "role": "button",
            "name": "Click me",
            "description": "A button",
            "focused": True,
            "children": [
                {"role": "text", "name": "Click me", "description": ""}
            ],
        }

        node = A11yNode.from_playwright(pw_data)

        assert node.role == "button"
        assert node.name == "Click me"
        assert node.focused is True
        assert len(node.children) == 1

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        node = A11yNode(
            role="button",
            name="Submit",
            description="",
            checked=True,
        )

        data = node.to_dict()

        assert data["role"] == "button"
        assert data["name"] == "Submit"
        assert data["checked"] is True


class TestA11yTree:
    """Tests for A11yTree dataclass."""

    def test_empty_tree(self) -> None:
        """Test creating empty tree."""
        tree = A11yTree()

        assert tree.root is None
        assert tree.node_count == 0

    def test_tree_with_root(self) -> None:
        """Test creating tree with root node."""
        root = A11yNode(
            role="document",
            name="Main page",
            description="",
            children=[
                A11yNode(role="button", name="Submit", description=""),
                A11yNode(role="link", name="Home", description=""),
            ],
        )

        tree = A11yTree(root=root)

        # Should count all nodes
        assert tree.node_count == 3

    def test_find_by_name(self) -> None:
        """Test finding nodes by name."""
        root = A11yNode(
            role="document",
            name="Main page",
            description="",
            children=[
                A11yNode(role="button", name="Submit", description=""),
                A11yNode(role="button", name="Cancel", description=""),
            ],
        )

        tree = A11yTree(root=root)

        # Find by exact name (case-insensitive)
        results = tree.find_by_name("Submit")
        assert len(results) == 1
        assert results[0].role == "button"

        results = tree.find_by_name("submit")  # lowercase
        assert len(results) == 1

    def test_find_by_role(self) -> None:
        """Test finding nodes by role."""
        root = A11yNode(
            role="document",
            name="Main page",
            description="",
            children=[
                A11yNode(role="button", name="Submit", description=""),
                A11yNode(role="button", name="Cancel", description=""),
                A11yNode(role="link", name="Home", description=""),
            ],
        )

        tree = A11yTree(root=root)

        buttons = tree.find_by_role("button")
        assert len(buttons) == 2

        links = tree.find_by_role("link")
        assert len(links) == 1

    def test_to_text(self) -> None:
        """Test text representation."""
        root = A11yNode(
            role="document",
            name="Main page",
            description="",
            children=[
                A11yNode(
                    role="button",
                    name="Submit",
                    description="Submit the form",
                    disabled=False,
                ),
            ],
        )

        tree = A11yTree(root=root)
        text = tree.to_text()

        assert "document" in text
        assert '"Main page"' in text
        assert "button" in text
        assert '"Submit"' in text

    def test_empty_tree_to_text(self) -> None:
        """Test text representation of empty tree."""
        tree = A11yTree()
        text = tree.to_text()

        assert text == "(empty)"


class TestAccessibilityExtractor:
    """Tests for AccessibilityExtractor class."""

    def test_roles_compatible(self) -> None:
        """Test role compatibility checking."""
        extractor = AccessibilityExtractor()

        # Button element should match button role
        element = InteractiveElement(
            id="btn1",
            bbox=BoundingBox(x=0, y=0, width=100, height=50),
            tag_name="button",
            element_type="button",
            screenshot_id="test",
            selector="button",
        )
        node = A11yNode(role="button", name="Submit", description="")

        assert extractor._roles_compatible(element, node) is True

        # Button element should not match link role
        node_link = A11yNode(role="link", name="Submit", description="")
        assert extractor._roles_compatible(element, node_link) is False

    def test_fuzzy_name_match(self) -> None:
        """Test fuzzy name matching."""
        extractor = AccessibilityExtractor()

        element = InteractiveElement(
            id="btn1",
            bbox=BoundingBox(x=0, y=0, width=100, height=50),
            tag_name="button",
            element_type="button",
            screenshot_id="test",
            selector="button",
            text="Submit Form",
        )

        # Exact match
        node = A11yNode(role="button", name="Submit Form", description="")
        assert extractor._fuzzy_name_match(element, node) is True

        # Partial match
        node_partial = A11yNode(role="button", name="Submit", description="")
        assert extractor._fuzzy_name_match(element, node_partial) is True

        # No match
        node_no_match = A11yNode(role="button", name="Cancel", description="")
        assert extractor._fuzzy_name_match(element, node_no_match) is False

    def test_match_element_to_a11y(self) -> None:
        """Test matching element to accessibility node."""
        extractor = AccessibilityExtractor()

        element = InteractiveElement(
            id="btn1",
            bbox=BoundingBox(x=0, y=0, width=100, height=50),
            tag_name="button",
            element_type="button",
            screenshot_id="test",
            selector="button",
            text="Submit",
        )

        root = A11yNode(
            role="document",
            name="Main",
            description="",
            children=[
                A11yNode(role="button", name="Submit", description="Submit form"),
            ],
        )
        tree = A11yTree(root=root)

        matched_node, confidence = extractor.match_element_to_a11y(element, tree)

        assert matched_node is not None
        assert matched_node.role == "button"
        assert confidence > 0


class TestEnrichedElement:
    """Tests for EnrichedElement dataclass."""

    def test_create_enriched_element(self) -> None:
        """Test creating an enriched element."""
        element = InteractiveElement(
            id="btn1",
            bbox=BoundingBox(x=0, y=0, width=100, height=50),
            tag_name="button",
            element_type="button",
            screenshot_id="test",
            selector="button",
            text="Submit",
        )

        enriched = EnrichedElement(
            element=element,
            a11y_role="button",
            a11y_name="Submit Form",
            a11y_description="Submits the registration form",
            match_confidence=0.95,
        )

        assert enriched.element.text == "Submit"
        assert enriched.a11y_role == "button"
        assert enriched.a11y_name == "Submit Form"
        assert enriched.match_confidence == 0.95

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        element = InteractiveElement(
            id="btn1",
            bbox=BoundingBox(x=0, y=0, width=100, height=50),
            tag_name="button",
            element_type="button",
            screenshot_id="test",
            selector="button",
        )

        enriched = EnrichedElement(
            element=element,
            a11y_role="button",
            match_confidence=0.9,
        )

        data = enriched.to_dict()

        assert "element" in data
        assert data["a11y_role"] == "button"
        assert data["match_confidence"] == 0.9


class TestA11yTreeToText:
    """Tests for a11y_tree_to_text convenience function."""

    def test_tree_to_text(self) -> None:
        """Test converting tree to text."""
        root = A11yNode(
            role="document",
            name="Page",
            description="",
            children=[
                A11yNode(role="button", name="Click", description=""),
            ],
        )
        tree = A11yTree(root=root)

        text = a11y_tree_to_text(tree)

        assert "document" in text
        assert "button" in text
        assert "Click" in text
