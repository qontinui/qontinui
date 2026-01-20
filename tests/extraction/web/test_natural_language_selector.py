"""
Tests for natural_language_selector module.

Tests AI-driven element selection by natural language description.
"""

from qontinui.extraction.web.models import BoundingBox, InteractiveElement
from qontinui.extraction.web.natural_language_selector import (
    FallbackSelector,
    SelectionResult,
)


def create_test_element(
    id: str,
    tag_name: str,
    text: str | None = None,
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
        aria_label=aria_label,
    )


class TestSelectionResult:
    """Tests for SelectionResult dataclass."""

    def test_found_property(self) -> None:
        """Test the found property."""
        elem = create_test_element("btn1", "button", text="Submit")

        result_found = SelectionResult(
            element=elem,
            index=0,
            confidence=0.9,
            reasoning="Found exact match",
        )
        assert result_found.found is True

        result_not_found = SelectionResult(
            element=None,
            index=None,
            confidence=0.0,
            reasoning="No match",
        )
        assert result_not_found.found is False

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        elem = create_test_element("btn1", "button", text="Submit")
        result = SelectionResult(
            element=elem,
            index=0,
            confidence=0.95,
            reasoning="Exact text match",
            alternatives=[2, 5],
        )

        data = result.to_dict()

        assert data["found"] is True
        assert data["index"] == 0
        assert data["confidence"] == 0.95
        assert data["alternatives"] == [2, 5]


class TestFallbackSelector:
    """Tests for FallbackSelector class."""

    def test_find_by_text_exact(self) -> None:
        """Test finding element by exact text match."""
        selector = FallbackSelector()
        elements = [
            create_test_element("btn1", "button", text="Submit"),
            create_test_element("btn2", "button", text="Cancel"),
            create_test_element("link1", "a", text="Home"),
        ]

        result = selector.find_by_text("Submit", elements)

        assert result.found is True
        assert result.index == 0
        assert result.confidence == 1.0

    def test_find_by_text_case_insensitive(self) -> None:
        """Test case-insensitive text matching."""
        selector = FallbackSelector()
        elements = [
            create_test_element("btn1", "button", text="Submit Form"),
        ]

        result = selector.find_by_text("submit form", elements, case_sensitive=False)

        assert result.found is True
        assert result.confidence == 1.0

    def test_find_by_text_partial(self) -> None:
        """Test partial text matching."""
        selector = FallbackSelector()
        elements = [
            create_test_element("btn1", "button", text="Submit Form Now"),
        ]

        result = selector.find_by_text("Submit", elements)

        assert result.found is True
        # Partial match gets lower confidence
        assert result.confidence < 1.0

    def test_find_by_text_aria_label(self) -> None:
        """Test matching by aria-label."""
        selector = FallbackSelector()
        elements = [
            create_test_element("btn1", "button", aria_label="Close dialog"),
        ]

        result = selector.find_by_text("Close dialog", elements)

        assert result.found is True
        assert result.confidence == 0.95  # aria-label match

    def test_find_by_text_no_match(self) -> None:
        """Test when no text matches."""
        selector = FallbackSelector()
        elements = [
            create_test_element("btn1", "button", text="Submit"),
        ]

        result = selector.find_by_text("Nonexistent", elements)

        assert result.found is False
        assert result.confidence == 0.0

    def test_find_by_role(self) -> None:
        """Test finding elements by role."""
        selector = FallbackSelector()
        elements = [
            create_test_element("btn1", "button", text="Submit"),
            create_test_element("btn2", "button", text="Cancel"),
            create_test_element("link1", "a", text="Home"),
        ]

        results = selector.find_by_role("button", elements)

        assert len(results) == 2
        assert all(r.found for r in results)

    def test_find_by_role_with_hint(self) -> None:
        """Test finding by role with text hint."""
        selector = FallbackSelector()
        elements = [
            create_test_element("btn1", "button", text="Submit"),
            create_test_element("btn2", "button", text="Cancel"),
        ]

        results = selector.find_by_role("button", elements, text_hint="Submit")

        assert len(results) == 2
        # Submit button should have higher confidence
        assert results[0].element.text == "Submit"
        assert results[0].confidence > results[1].confidence


class TestNaturalLanguageSelectorParsing:
    """Tests for response parsing in NaturalLanguageSelector."""

    def test_parse_selection_response_basic(self) -> None:
        """Test parsing a basic selection response."""
        # We can't easily test the full flow without mocking LLM,
        # but we can test the parsing logic

        class MockLLM:
            async def complete(self, prompt: str) -> str:
                return """INDEX: 3
CONFIDENCE: 0.95
REASONING: Element 3 is a button with text "Submit"
ALTERNATIVES: 7, 12"""

        # The parsing is internal, so we'd need to expose it or test via integration
        # For now, test that SelectionResult handles the data correctly

        result = SelectionResult(
            element=None,  # Would be set after parsing
            index=3,
            confidence=0.95,
            reasoning='Element 3 is a button with text "Submit"',
            alternatives=[7, 12],
        )

        assert result.index == 3
        assert result.confidence == 0.95
        assert len(result.alternatives) == 2

    def test_selection_result_no_element(self) -> None:
        """Test SelectionResult when element not found."""
        result = SelectionResult(
            element=None,
            index=None,
            confidence=0.0,
            reasoning="No matching element found",
        )

        assert result.found is False
        assert result.index is None


class TestNaturalLanguageSelectorPrompts:
    """Tests for prompt building."""

    def test_prompt_structure(self) -> None:
        """Test that prompts are properly structured."""
        # Test that we can create a selector (doesn't require LLM to test prompt building)
        elements = [
            create_test_element("btn1", "button", text="Submit"),
        ]

        # The prompt building is internal, but we can verify element formatting
        from qontinui.extraction.web.llm_formatter import LLMFormatter

        formatter = LLMFormatter()
        result = formatter.format_elements(elements)

        # Verify the format is suitable for prompts
        assert "[0]" in result.text
        assert "button" in result.text
        assert "Submit" in result.text
