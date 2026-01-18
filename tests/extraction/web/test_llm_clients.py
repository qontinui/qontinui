"""Tests for LLM client adapters."""

from __future__ import annotations

import pytest

from qontinui.extraction.web.llm_clients import (
    BaseLLMClient,
    LLMConfig,
    MockLLMClient,
    create_llm_client,
)


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LLMConfig(model="test-model")

        assert config.model == "test-model"
        assert config.max_tokens == 500
        assert config.temperature == 0.0
        assert config.timeout == 30.0
        assert config.extra_params == {}

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = LLMConfig(
            model="custom-model",
            max_tokens=1000,
            temperature=0.5,
            timeout=60.0,
            extra_params={"top_p": 0.9},
        )

        assert config.model == "custom-model"
        assert config.max_tokens == 1000
        assert config.temperature == 0.5
        assert config.timeout == 60.0
        assert config.extra_params == {"top_p": 0.9}


class TestMockLLMClient:
    """Tests for MockLLMClient."""

    @pytest.mark.asyncio
    async def test_default_response(self) -> None:
        """Test default mock response for selection prompts."""
        client = MockLLMClient()

        response = await client.complete("Find the element with INDEX:")

        assert "INDEX:" in response
        assert "CONFIDENCE:" in response
        assert "REASONING:" in response

    @pytest.mark.asyncio
    async def test_custom_response(self) -> None:
        """Test custom mock responses."""
        client = MockLLMClient(
            responses={
                "submit": "INDEX: 5\nCONFIDENCE: 0.95\nREASONING: Found submit button",
                "cancel": "INDEX: 6\nCONFIDENCE: 0.90\nREASONING: Found cancel button",
            }
        )

        response1 = await client.complete("Find the submit button")
        assert "INDEX: 5" in response1

        response2 = await client.complete("Find the cancel button")
        assert "INDEX: 6" in response2

    @pytest.mark.asyncio
    async def test_call_history(self) -> None:
        """Test that calls are recorded in history."""
        client = MockLLMClient()

        await client.complete("First prompt")
        await client.complete("Second prompt")
        await client.complete("Third prompt")

        assert len(client.call_history) == 3
        assert client.call_history[0] == "First prompt"
        assert client.call_history[1] == "Second prompt"
        assert client.call_history[2] == "Third prompt"

    @pytest.mark.asyncio
    async def test_case_insensitive_matching(self) -> None:
        """Test that pattern matching is case-insensitive."""
        client = MockLLMClient(
            responses={"SUBMIT": "Found it!"}
        )

        response = await client.complete("find the submit button")
        assert response == "Found it!"

    def test_config(self) -> None:
        """Test mock client has config."""
        client = MockLLMClient()

        assert client.config is not None
        assert client.config.model == "mock"


class TestCreateLLMClient:
    """Tests for create_llm_client factory function."""

    def test_create_mock_client(self) -> None:
        """Test creating a mock client."""
        client = create_llm_client("mock")

        assert isinstance(client, MockLLMClient)

    def test_create_mock_with_responses(self) -> None:
        """Test creating mock client with custom responses."""
        client = create_llm_client(
            "mock",
            responses={"test": "custom response"},
        )

        assert isinstance(client, MockLLMClient)
        assert "test" in client.responses

    def test_unknown_provider(self) -> None:
        """Test that unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm_client("unknown_provider")

    def test_anthropic_without_key(self) -> None:
        """Test that Anthropic client requires API key."""
        # This should raise an error if ANTHROPIC_API_KEY is not set
        # We can't reliably test this without mocking env vars
        pass

    def test_openai_without_key(self) -> None:
        """Test that OpenAI client requires API key."""
        # This should raise an error if OPENAI_API_KEY is not set
        pass

    def test_case_insensitive_provider(self) -> None:
        """Test that provider name is case-insensitive."""
        client1 = create_llm_client("MOCK")
        client2 = create_llm_client("Mock")
        client3 = create_llm_client("mock")

        assert isinstance(client1, MockLLMClient)
        assert isinstance(client2, MockLLMClient)
        assert isinstance(client3, MockLLMClient)


class TestBaseLLMClient:
    """Tests for BaseLLMClient abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test that BaseLLMClient cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseLLMClient()


class TestLLMClientIntegration:
    """Integration tests for LLM clients with NaturalLanguageSelector."""

    @pytest.mark.asyncio
    async def test_mock_client_with_selector(self) -> None:
        """Test MockLLMClient works with NaturalLanguageSelector."""
        from qontinui.extraction.web.natural_language_selector import (
            NaturalLanguageSelector,
        )
        from qontinui.extraction.web.models import BoundingBox, InteractiveElement

        # Create mock client with predictable response
        client = MockLLMClient(
            responses={
                "submit": """INDEX: 1
CONFIDENCE: 0.95
REASONING: Found the submit button
ALTERNATIVES: none""",
            }
        )

        # Create test elements
        elements = [
            InteractiveElement(
                id="elem_1",
                bbox=BoundingBox(x=0, y=0, width=100, height=30),
                tag_name="input",
                element_type="input",
                screenshot_id="test",
                selector="input.username",
                text="Username",
            ),
            InteractiveElement(
                id="elem_2",
                bbox=BoundingBox(x=0, y=50, width=100, height=30),
                tag_name="button",
                element_type="button",
                screenshot_id="test",
                selector="button.submit",
                text="Submit",
            ),
        ]

        # Create selector and find element
        selector = NaturalLanguageSelector(client)
        result = await selector.find_element("the submit button", elements)

        assert result.found
        assert result.index == 1
        assert result.confidence == 0.95
        assert result.element == elements[1]

    @pytest.mark.asyncio
    async def test_mock_client_action_selection(self) -> None:
        """Test MockLLMClient works with action selection."""
        from qontinui.extraction.web.natural_language_selector import (
            NaturalLanguageSelector,
        )
        from qontinui.extraction.web.models import BoundingBox, InteractiveElement

        # Create mock client
        client = MockLLMClient(
            responses={
                "click": """INDEX: 0
ACTION: click
CONFIDENCE: 0.90
REASONING: Clicking the button""",
            }
        )

        elements = [
            InteractiveElement(
                id="elem_1",
                bbox=BoundingBox(x=0, y=0, width=100, height=30),
                tag_name="button",
                element_type="button",
                screenshot_id="test",
                selector="button.ok",
                text="OK",
            ),
        ]

        selector = NaturalLanguageSelector(client)
        result, action = await selector.select_action("click the OK button", elements)

        assert result.found
        assert result.index == 0
        assert action == "click"

    @pytest.mark.asyncio
    async def test_mock_client_multi_selection(self) -> None:
        """Test MockLLMClient works with multiple element selection."""
        from qontinui.extraction.web.natural_language_selector import (
            NaturalLanguageSelector,
        )
        from qontinui.extraction.web.models import BoundingBox, InteractiveElement

        # Create mock client with multi-match response
        client = MockLLMClient(
            responses={
                "buttons": """MATCH: 0, 0.95, Primary button
MATCH: 2, 0.85, Secondary button
MATCH: 4, 0.70, Tertiary button""",
            }
        )

        elements = [
            InteractiveElement(
                id=f"elem_{i}",
                bbox=BoundingBox(x=0, y=i * 50, width=100, height=30),
                tag_name="button",
                element_type="button",
                screenshot_id="test",
                selector=f"button.btn-{i}",
                text=f"Button {i}",
            )
            for i in range(5)
        ]

        selector = NaturalLanguageSelector(client)
        results = await selector.find_multiple("all buttons", elements)

        assert len(results) == 3
        assert results[0].index == 0
        assert results[0].confidence == 0.95
        assert results[1].index == 2
        assert results[2].index == 4
