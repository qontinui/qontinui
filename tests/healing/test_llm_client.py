"""Tests for LLM client implementations in the healing system.

Tests cover:
- DisabledVisionClient (always returns None)
- LocalVisionClient (Ollama) with mocked HTTP responses
- RemoteVisionClient (OpenAI, Anthropic, Google) with mocked API responses
- Response parsing for coordinate extraction
- Error handling and availability checks
"""

import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from qontinui.healing.healing_types import HealingContext, ElementLocation
from qontinui.healing.llm_client import (
    DisabledVisionClient,
    LocalVisionClient,
    RemoteVisionClient,
    VisionLLMClient,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_screenshot() -> bytes:
    """Create a minimal PNG image for testing."""
    # 1x1 white PNG
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )


@pytest.fixture
def basic_context() -> HealingContext:
    """Create a basic HealingContext for testing."""
    return HealingContext(
        original_description="Submit button",
        action_type="click",
        failure_reason="Element not found",
    )


@pytest.fixture
def context_without_extras() -> HealingContext:
    """Create a minimal HealingContext."""
    return HealingContext(original_description="OK button")


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client with context manager support."""
    mock_client = MagicMock()
    mock_context = MagicMock()
    mock_context.__enter__ = MagicMock(return_value=mock_client)
    mock_context.__exit__ = MagicMock(return_value=None)
    return mock_context, mock_client


# ============================================================================
# DisabledVisionClient Tests
# ============================================================================


class TestDisabledVisionClient:
    """Tests for DisabledVisionClient."""

    def test_always_returns_none(self, sample_screenshot, basic_context):
        """Disabled client always returns None."""
        client = DisabledVisionClient()
        result = client.find_element(sample_screenshot, basic_context)
        assert result is None

    def test_is_always_available(self):
        """Disabled client is always available."""
        client = DisabledVisionClient()
        assert client.is_available is True

    def test_works_with_empty_screenshot(self, basic_context):
        """Disabled client handles empty screenshot."""
        client = DisabledVisionClient()
        result = client.find_element(b"", basic_context)
        assert result is None

    def test_works_with_minimal_context(self, sample_screenshot, context_without_extras):
        """Disabled client works with minimal context."""
        client = DisabledVisionClient()
        result = client.find_element(sample_screenshot, context_without_extras)
        assert result is None


# ============================================================================
# Response Parsing Tests
# ============================================================================


class TestResponseParsing:
    """Tests for _parse_response method shared by all clients."""

    def test_parse_coordinates_standard_format(self):
        """Parse standard COORDINATES format."""
        client = DisabledVisionClient()
        result = client._parse_response("COORDINATES: 450,320")

        assert result is not None
        assert result.x == 450
        assert result.y == 320
        assert result.confidence == 0.8

    def test_parse_coordinates_with_spaces(self):
        """Parse coordinates with extra spaces."""
        client = DisabledVisionClient()
        result = client._parse_response("COORDINATES:   100 ,  200  ")

        assert result is not None
        assert result.x == 100
        assert result.y == 200

    def test_parse_coordinates_lowercase(self):
        """Parse lowercase coordinates format."""
        client = DisabledVisionClient()
        result = client._parse_response("coordinates: 300,400")

        assert result is not None
        assert result.x == 300
        assert result.y == 400

    def test_parse_simple_format(self):
        """Parse simple x,y format."""
        client = DisabledVisionClient()
        result = client._parse_response("500,600")

        assert result is not None
        assert result.x == 500
        assert result.y == 600
        assert result.confidence == 0.7  # Lower confidence for simple format

    def test_parse_not_found(self):
        """Parse NOT_FOUND response."""
        client = DisabledVisionClient()
        result = client._parse_response("NOT_FOUND: element is not visible")

        assert result is None

    def test_parse_not_found_uppercase(self):
        """Parse NOT_FOUND in various cases."""
        client = DisabledVisionClient()

        assert client._parse_response("NOT_FOUND: reason") is None
        assert client._parse_response("not_found: reason") is None
        assert client._parse_response("Not_Found: reason") is None

    def test_parse_invalid_response(self):
        """Parse invalid response returns None."""
        client = DisabledVisionClient()

        assert client._parse_response("I found the element") is None
        assert client._parse_response("The button is at position 450, 320") is None
        assert client._parse_response("") is None
        assert client._parse_response("random text") is None

    def test_parse_coordinates_in_text(self):
        """Parse coordinates embedded in other text."""
        client = DisabledVisionClient()
        result = client._parse_response(
            "I analyzed the image. COORDINATES: 150,250. That's where the button is."
        )

        assert result is not None
        assert result.x == 150
        assert result.y == 250


# ============================================================================
# Prompt Building Tests
# ============================================================================


class TestPromptBuilding:
    """Tests for _build_prompt method."""

    def test_prompt_includes_description(self, basic_context):
        """Prompt includes element description."""
        client = DisabledVisionClient()
        prompt = client._build_prompt(basic_context)

        assert "Submit button" in prompt
        assert "Element to find:" in prompt

    def test_prompt_includes_action_type(self, basic_context):
        """Prompt includes action type when provided."""
        client = DisabledVisionClient()
        prompt = client._build_prompt(basic_context)

        assert "click" in prompt
        assert "Action intended:" in prompt

    def test_prompt_includes_failure_reason(self, basic_context):
        """Prompt includes failure reason when provided."""
        client = DisabledVisionClient()
        prompt = client._build_prompt(basic_context)

        assert "Element not found" in prompt
        assert "Previous lookup failed" in prompt

    def test_prompt_minimal_context(self, context_without_extras):
        """Prompt works with minimal context."""
        client = DisabledVisionClient()
        prompt = client._build_prompt(context_without_extras)

        assert "OK button" in prompt
        assert "Action intended:" not in prompt
        assert "Previous lookup failed" not in prompt

    def test_prompt_includes_output_format(self, basic_context):
        """Prompt includes expected output format."""
        client = DisabledVisionClient()
        prompt = client._build_prompt(basic_context)

        assert "COORDINATES:" in prompt
        assert "NOT_FOUND:" in prompt


# ============================================================================
# LocalVisionClient (Ollama) Tests
# ============================================================================


class TestLocalVisionClient:
    """Tests for LocalVisionClient (Ollama)."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        client = LocalVisionClient()

        assert client.model_name == "llava:7b"
        assert client.base_url == "http://localhost:11434"
        assert client.timeout_seconds == 30.0

    def test_initialization_custom_values(self):
        """Test custom initialization values."""
        client = LocalVisionClient(
            model_name="bakllava",
            base_url="http://custom:8080",
            timeout_seconds=60.0,
        )

        assert client.model_name == "bakllava"
        assert client.base_url == "http://custom:8080"
        assert client.timeout_seconds == 60.0

    def test_base_url_trailing_slash_stripped(self):
        """Base URL trailing slash is stripped."""
        client = LocalVisionClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"

    def test_find_element_success(self, sample_screenshot, basic_context):
        """Test successful element finding via Ollama."""
        # Create mock httpx module
        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_ctx.__exit__ = MagicMock(return_value=None)
        mock_httpx.Client.return_value = mock_client_ctx

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "COORDINATES: 100,200"}
        mock_response.raise_for_status = MagicMock()
        mock_client_instance.post.return_value = mock_response

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            client = LocalVisionClient()
            result = client.find_element(sample_screenshot, basic_context)

        assert result is not None
        assert result.x == 100
        assert result.y == 200
        assert result.confidence == 0.8

    def test_find_element_not_found(self, sample_screenshot, basic_context):
        """Test Ollama returns NOT_FOUND."""
        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_ctx.__exit__ = MagicMock(return_value=None)
        mock_httpx.Client.return_value = mock_client_ctx

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "NOT_FOUND: no button visible"}
        mock_response.raise_for_status = MagicMock()
        mock_client_instance.post.return_value = mock_response

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            client = LocalVisionClient()
            result = client.find_element(sample_screenshot, basic_context)

        assert result is None

    def test_find_element_http_error(self, sample_screenshot, basic_context):
        """Test HTTP error handling."""
        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_ctx.__exit__ = MagicMock(return_value=None)
        mock_httpx.Client.return_value = mock_client_ctx
        mock_httpx.HTTPError = Exception

        mock_client_instance.post.side_effect = mock_httpx.HTTPError("Connection refused")

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            client = LocalVisionClient()
            result = client.find_element(sample_screenshot, basic_context)

        assert result is None

    def test_find_element_httpx_not_installed(self, sample_screenshot, basic_context):
        """Test when httpx is not installed."""
        # Simulate httpx import failure
        with patch.dict(sys.modules, {"httpx": None}):
            client = LocalVisionClient()
            # The import will fail inside find_element, which should return None
            result = client.find_element(sample_screenshot, basic_context)
            # Should gracefully return None
            assert result is None

    def test_is_available_true(self):
        """Test availability check when Ollama is running."""
        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_ctx.__exit__ = MagicMock(return_value=None)
        mock_httpx.Client.return_value = mock_client_ctx

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llava:7b"}, {"name": "llama2:13b"}]
        }
        mock_client_instance.get.return_value = mock_response

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            client = LocalVisionClient(model_name="llava:7b")
            assert client.is_available is True

    def test_is_available_model_not_found(self):
        """Test availability when model is not installed."""
        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_ctx.__exit__ = MagicMock(return_value=None)
        mock_httpx.Client.return_value = mock_client_ctx

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama2:7b"}]  # Different model
        }
        mock_client_instance.get.return_value = mock_response

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            client = LocalVisionClient(model_name="llava:7b")
            assert client.is_available is False

    def test_is_available_ollama_not_running(self):
        """Test availability when Ollama is not running."""
        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_ctx.__exit__ = MagicMock(return_value=None)
        mock_httpx.Client.return_value = mock_client_ctx

        mock_client_instance.get.side_effect = Exception("Connection refused")

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            client = LocalVisionClient()
            assert client.is_available is False

    def test_is_available_non_200_status(self):
        """Test availability with non-200 status code."""
        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_ctx.__exit__ = MagicMock(return_value=None)
        mock_httpx.Client.return_value = mock_client_ctx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client_instance.get.return_value = mock_response

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            client = LocalVisionClient()
            assert client.is_available is False


# ============================================================================
# RemoteVisionClient Tests - OpenAI
# ============================================================================


class TestRemoteVisionClientOpenAI:
    """Tests for RemoteVisionClient with OpenAI provider."""

    def test_initialization_openai(self):
        """Test OpenAI initialization."""
        client = RemoteVisionClient(
            provider="openai",
            api_key="sk-test-key",
        )

        assert client.provider == "openai"
        assert client.api_key == "sk-test-key"
        assert client.model == "gpt-4o"  # Default model

    def test_initialization_custom_model(self):
        """Test OpenAI with custom model."""
        client = RemoteVisionClient(
            provider="openai",
            api_key="sk-test-key",
            model="gpt-4-vision-preview",
        )

        assert client.model == "gpt-4-vision-preview"

    def test_find_element_success(self, sample_screenshot, basic_context):
        """Test successful element finding via OpenAI."""
        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_ctx.__exit__ = MagicMock(return_value=None)
        mock_httpx.Client.return_value = mock_client_ctx

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "COORDINATES: 300,400"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client_instance.post.return_value = mock_response

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            client = RemoteVisionClient(provider="openai", api_key="sk-test")
            result = client.find_element(sample_screenshot, basic_context)

        assert result is not None
        assert result.x == 300
        assert result.y == 400

        # Verify correct endpoint called
        call_args = mock_client_instance.post.call_args
        assert "chat/completions" in call_args[0][0]

    def test_find_element_api_error(self, sample_screenshot, basic_context):
        """Test OpenAI API error handling."""
        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_ctx.__exit__ = MagicMock(return_value=None)
        mock_httpx.Client.return_value = mock_client_ctx
        mock_httpx.HTTPError = Exception

        mock_client_instance.post.side_effect = mock_httpx.HTTPError("Rate limit exceeded")

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            client = RemoteVisionClient(provider="openai", api_key="sk-test")
            result = client.find_element(sample_screenshot, basic_context)

        assert result is None

    def test_find_element_invalid_response_format(
        self, sample_screenshot, basic_context
    ):
        """Test OpenAI with unexpected response format."""
        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_ctx.__exit__ = MagicMock(return_value=None)
        mock_httpx.Client.return_value = mock_client_ctx
        # Set HTTPError to a real exception class for proper exception handling
        mock_httpx.HTTPError = Exception

        mock_response = MagicMock()
        mock_response.json.return_value = {"unexpected": "format"}
        mock_response.raise_for_status = MagicMock()
        mock_client_instance.post.return_value = mock_response

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            client = RemoteVisionClient(provider="openai", api_key="sk-test")
            result = client.find_element(sample_screenshot, basic_context)

        assert result is None

    def test_is_available_with_api_key(self):
        """Test availability with API key."""
        client = RemoteVisionClient(provider="openai", api_key="sk-test")
        assert client.is_available is True

    def test_is_available_without_api_key(self):
        """Test availability without API key."""
        client = RemoteVisionClient(provider="openai", api_key="")
        assert client.is_available is False

    def test_custom_base_url(self, sample_screenshot, basic_context):
        """Test OpenAI with custom base URL."""
        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_ctx.__exit__ = MagicMock(return_value=None)
        mock_httpx.Client.return_value = mock_client_ctx

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "COORDINATES: 100,100"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client_instance.post.return_value = mock_response

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            client = RemoteVisionClient(
                provider="openai",
                api_key="sk-test",
                base_url="https://custom-openai.example.com/v1",
            )
            client.find_element(sample_screenshot, basic_context)

        # Verify custom URL was used
        call_args = mock_client_instance.post.call_args
        assert "custom-openai.example.com" in call_args[0][0]


# ============================================================================
# RemoteVisionClient Tests - Anthropic
# ============================================================================


class TestRemoteVisionClientAnthropic:
    """Tests for RemoteVisionClient with Anthropic provider."""

    def test_initialization_anthropic(self):
        """Test Anthropic initialization."""
        client = RemoteVisionClient(
            provider="anthropic",
            api_key="sk-ant-test",
        )

        assert client.provider == "anthropic"
        assert client.api_key == "sk-ant-test"
        assert client.model == "claude-sonnet-4-20250514"  # Default model

    def test_find_element_success(self, sample_screenshot, basic_context):
        """Test successful element finding via Anthropic."""
        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_ctx.__exit__ = MagicMock(return_value=None)
        mock_httpx.Client.return_value = mock_client_ctx

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"text": "COORDINATES: 500,600"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client_instance.post.return_value = mock_response

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            client = RemoteVisionClient(provider="anthropic", api_key="sk-ant-test")
            result = client.find_element(sample_screenshot, basic_context)

        assert result is not None
        assert result.x == 500
        assert result.y == 600

        # Verify correct endpoint called
        call_args = mock_client_instance.post.call_args
        assert "messages" in call_args[0][0]

    def test_find_element_with_headers(
        self, sample_screenshot, basic_context
    ):
        """Test Anthropic request includes correct headers."""
        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_ctx.__exit__ = MagicMock(return_value=None)
        mock_httpx.Client.return_value = mock_client_ctx

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"text": "COORDINATES: 100,100"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client_instance.post.return_value = mock_response

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            client = RemoteVisionClient(provider="anthropic", api_key="sk-ant-test")
            client.find_element(sample_screenshot, basic_context)

        # Verify headers
        call_kwargs = mock_client_instance.post.call_args[1]
        headers = call_kwargs.get("headers", {})
        assert "x-api-key" in headers
        assert "anthropic-version" in headers

    def test_find_element_api_error(self, sample_screenshot, basic_context):
        """Test Anthropic API error handling."""
        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_ctx.__exit__ = MagicMock(return_value=None)
        mock_httpx.Client.return_value = mock_client_ctx
        mock_httpx.HTTPError = Exception

        mock_client_instance.post.side_effect = mock_httpx.HTTPError("Unauthorized")

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            client = RemoteVisionClient(provider="anthropic", api_key="sk-ant-test")
            result = client.find_element(sample_screenshot, basic_context)

        assert result is None


# ============================================================================
# RemoteVisionClient Tests - Google
# ============================================================================


class TestRemoteVisionClientGoogle:
    """Tests for RemoteVisionClient with Google provider."""

    def test_initialization_google(self):
        """Test Google initialization."""
        client = RemoteVisionClient(
            provider="google",
            api_key="google-api-key",
        )

        assert client.provider == "google"
        assert client.api_key == "google-api-key"
        assert client.model == "gemini-1.5-flash"  # Default model

    def test_find_element_success(self, sample_screenshot, basic_context):
        """Test successful element finding via Google."""
        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_ctx.__exit__ = MagicMock(return_value=None)
        mock_httpx.Client.return_value = mock_client_ctx

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "COORDINATES: 750,850"}]}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client_instance.post.return_value = mock_response

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            client = RemoteVisionClient(provider="google", api_key="google-key")
            result = client.find_element(sample_screenshot, basic_context)

        assert result is not None
        assert result.x == 750
        assert result.y == 850

    def test_find_element_api_key_in_params(
        self, sample_screenshot, basic_context
    ):
        """Test Google API key is passed in params."""
        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_ctx.__exit__ = MagicMock(return_value=None)
        mock_httpx.Client.return_value = mock_client_ctx

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "COORDINATES: 100,100"}]}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client_instance.post.return_value = mock_response

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            client = RemoteVisionClient(provider="google", api_key="google-key")
            client.find_element(sample_screenshot, basic_context)

        # Verify API key in params
        call_kwargs = mock_client_instance.post.call_args[1]
        params = call_kwargs.get("params", {})
        assert params.get("key") == "google-key"


# ============================================================================
# RemoteVisionClient Tests - Unknown Provider
# ============================================================================


class TestRemoteVisionClientUnknown:
    """Tests for RemoteVisionClient with unknown provider."""

    def test_unknown_provider_raises_error(self):
        """Test unknown provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            RemoteVisionClient(provider="unknown", api_key="test")

        assert "Unknown provider" in str(exc_info.value)

    def test_find_element_unknown_provider(self, sample_screenshot, basic_context):
        """Test find_element with unknown provider returns None."""
        # Create client with a valid provider, then change it
        client = RemoteVisionClient(provider="openai", api_key="test")
        client.provider = "unknown_provider"

        result = client.find_element(sample_screenshot, basic_context)
        assert result is None


# ============================================================================
# Integration Tests (with mocks)
# ============================================================================


class TestLLMClientIntegration:
    """Integration tests for LLM clients with HealingConfig."""

    def test_disabled_config_creates_disabled_client(self):
        """Test disabled config creates DisabledVisionClient."""
        from qontinui.healing.healing_config import HealingConfig

        config = HealingConfig.disabled()
        client = config.create_client()

        assert type(client).__name__ == "DisabledVisionClient"

    def test_ollama_config_creates_local_client(self):
        """Test Ollama config creates LocalVisionClient."""
        from qontinui.healing.healing_config import HealingConfig

        config = HealingConfig.with_ollama(model_name="bakllava")
        client = config.create_client()

        assert type(client).__name__ == "LocalVisionClient"
        assert client.model_name == "bakllava"

    def test_openai_config_creates_remote_client(self):
        """Test OpenAI config creates RemoteVisionClient."""
        from qontinui.healing.healing_config import HealingConfig

        config = HealingConfig.with_openai(api_key="sk-test", model="gpt-4o")
        client = config.create_client()

        assert type(client).__name__ == "RemoteVisionClient"
        assert client.provider == "openai"
        assert client.model == "gpt-4o"

    def test_anthropic_config_creates_remote_client(self):
        """Test Anthropic config creates RemoteVisionClient."""
        from qontinui.healing.healing_config import HealingConfig

        config = HealingConfig.with_anthropic(api_key="sk-ant-test")
        client = config.create_client()

        assert type(client).__name__ == "RemoteVisionClient"
        assert client.provider == "anthropic"

    def test_client_caching(self):
        """Test client is cached after first creation."""
        from qontinui.healing.healing_config import HealingConfig

        config = HealingConfig.disabled()
        client1 = config.get_client()
        client2 = config.get_client()

        assert client1 is client2


# ============================================================================
# Abstract Class Tests
# ============================================================================


class TestVisionLLMClientABC:
    """Tests for VisionLLMClient abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that VisionLLMClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VisionLLMClient()

    def test_subclass_must_implement_methods(self):
        """Test that subclasses must implement abstract methods."""

        class IncompleteClient(VisionLLMClient):
            pass

        with pytest.raises(TypeError):
            IncompleteClient()

    def test_complete_subclass_works(self):
        """Test that complete subclass can be instantiated."""

        class CompleteClient(VisionLLMClient):
            def find_element(self, screenshot, context):
                return None

            @property
            def is_available(self):
                return True

        client = CompleteClient()
        assert client.is_available is True
