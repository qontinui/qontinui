"""
Tests for llm_formatter module.

Tests LLM-friendly element formatting with numeric indices.
"""

from qontinui.extraction.web.llm_formatter import (
    LLMFormatter,
    format_for_llm,
    get_element_by_index,
    parse_llm_element_reference,
)
from qontinui.extraction.web.models import BoundingBox, InteractiveElement


def create_test_element(
    id: str,
    tag_name: str,
    text: str | None = None,
    href: str | None = None,
    aria_label: str | None = None,
    element_type: str | None = None,
) -> InteractiveElement:
    """Helper to create test elements."""
    return InteractiveElement(
        id=id,
        bbox=BoundingBox(x=0, y=0, width=100, height=50),
        tag_name=tag_name,
        element_type=element_type or tag_name,
        screenshot_id="test_screen",
        selector=f"{tag_name}#{id}",
        text=text,
        href=href,
        aria_label=aria_label,
    )


class TestLLMFormatter:
    """Tests for LLMFormatter class."""

    def test_format_button(self) -> None:
        """Test formatting a button element."""
        formatter = LLMFormatter()
        elements = [create_test_element("btn1", "button", text="Submit")]

        result = formatter.format_elements(elements)

        assert "[0]<button>Submit</button>" in result.text
        assert len(result.elements) == 1
        assert result.elements[0].index == 0

    def test_format_link(self) -> None:
        """Test formatting a link element."""
        formatter = LLMFormatter()
        elements = [create_test_element("link1", "a", text="Home", href="/home")]

        result = formatter.format_elements(elements)

        assert "[0]<a href=" in result.text
        assert "Home</a>" in result.text

    def test_format_multiple_elements(self) -> None:
        """Test formatting multiple elements."""
        formatter = LLMFormatter()
        elements = [
            create_test_element("btn1", "button", text="Submit"),
            create_test_element("link1", "a", text="Home", href="/home"),
            create_test_element("input1", "input", aria_label="Email"),
        ]

        result = formatter.format_elements(elements)

        assert "[0]<button>" in result.text
        assert "[1]<a" in result.text
        assert "[2]<input" in result.text

    def test_format_with_aria_label(self) -> None:
        """Test formatting element with aria-label."""
        formatter = LLMFormatter()
        elements = [create_test_element("btn1", "button", text="X", aria_label="Close dialog")]

        result = formatter.format_elements(elements)

        # aria-label should be included since it differs from text
        assert 'aria-label="Close dialog"' in result.text

    def test_truncate_long_text(self) -> None:
        """Test that long text is truncated."""
        formatter = LLMFormatter(max_text_length=20)
        elements = [
            create_test_element(
                "btn1", "button", text="This is a very long button text that should be truncated"
            )
        ]

        result = formatter.format_elements(elements)

        # Text should be truncated
        assert "..." in result.text
        assert len(result.elements[0].formatted) < 100

    def test_format_for_context(self) -> None:
        """Test formatting with page context."""
        formatter = LLMFormatter()
        elements = [create_test_element("btn1", "button", text="Submit")]

        result = formatter.format_for_context(
            elements,
            page_url="https://example.com",
            page_title="Example Page",
        )

        assert "Page: Example Page (https://example.com)" in result
        assert "Interactive Elements (1):" in result
        assert "[0]<button>" in result

    def test_get_by_index(self) -> None:
        """Test getting element by index from formatted list."""
        formatter = LLMFormatter()
        elements = [
            create_test_element("btn1", "button", text="Submit"),
            create_test_element("btn2", "button", text="Cancel"),
        ]

        result = formatter.format_elements(elements)

        elem = result.get_by_index(0)
        assert elem is not None
        assert elem.index == 0

        elem = result.get_by_index(1)
        assert elem is not None
        assert elem.index == 1

        elem = result.get_by_index(99)
        assert elem is None


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_format_for_llm(self) -> None:
        """Test format_for_llm convenience function."""
        elements = [create_test_element("btn1", "button", text="Submit")]

        result = format_for_llm(
            elements,
            page_url="https://example.com",
            page_title="Test",
        )

        assert "Page: Test" in result
        assert "[0]<button>" in result

    def test_get_element_by_index(self) -> None:
        """Test get_element_by_index convenience function."""
        elements = [
            create_test_element("btn1", "button", text="Submit"),
            create_test_element("btn2", "button", text="Cancel"),
        ]

        elem = get_element_by_index(elements, 0)
        assert elem is not None
        assert elem.text == "Submit"

        elem = get_element_by_index(elements, 1)
        assert elem is not None
        assert elem.text == "Cancel"

        elem = get_element_by_index(elements, 99)
        assert elem is None


class TestParseLLMElementReference:
    """Tests for parse_llm_element_reference function."""

    def test_parse_simple_number(self) -> None:
        """Test parsing simple number."""
        assert parse_llm_element_reference("5") == 5
        assert parse_llm_element_reference("0") == 0
        assert parse_llm_element_reference("123") == 123

    def test_parse_bracketed_number(self) -> None:
        """Test parsing [N] format."""
        assert parse_llm_element_reference("[5]") == 5
        assert parse_llm_element_reference("[0]") == 0
        assert parse_llm_element_reference("Click on [3]") == 3

    def test_parse_element_n_format(self) -> None:
        """Test parsing 'element N' format."""
        assert parse_llm_element_reference("element 5") == 5
        assert parse_llm_element_reference("Element 3") == 3
        assert parse_llm_element_reference("click element 7") == 7

    def test_parse_index_n_format(self) -> None:
        """Test parsing 'index N' format."""
        assert parse_llm_element_reference("index 5") == 5
        assert parse_llm_element_reference("Index 3") == 3

    def test_parse_number_in_sentence(self) -> None:
        """Test parsing number embedded in sentence."""
        assert parse_llm_element_reference("I would click 5") == 5
        assert parse_llm_element_reference("The answer is 3") == 3

    def test_parse_none_response(self) -> None:
        """Test parsing 'none' returns None."""
        # Only returns None if there's no number at all
        assert parse_llm_element_reference("none") is None
        assert parse_llm_element_reference("no match") is None

    def test_parse_empty_string(self) -> None:
        """Test parsing empty string returns None."""
        assert parse_llm_element_reference("") is None
        assert parse_llm_element_reference("   ") is None
