"""Tests for AriaUIClient and AriaUIContextClient."""

import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from qontinui.healing.aria_ui_client import AriaUIClient
from qontinui.healing.aria_ui_context_client import AriaUIContextClient
from qontinui.healing.healing_types import ElementLocation, HealingContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_SCREENSHOT = b"\x89PNG\r\n\x1a\nfake_png_data"


def _make_context(
    description: str = "Click the Submit button",
    screenshot_shape: tuple[int, int] | None = (1080, 1920),
    additional_context: dict | None = None,
) -> HealingContext:
    return HealingContext(
        original_description=description,
        screenshot_shape=screenshot_shape,
        additional_context=additional_context or {},
    )


def _make_chat_response(content: str) -> dict:
    """Build a minimal OpenAI-style chat completion response."""
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                }
            }
        ]
    }


class _MockHTTPError(Exception):
    """Stands in for httpx.HTTPError so except clauses work."""

    pass


def _make_mock_httpx():
    """Create a mock httpx module with Client context manager wired up.

    Returns (mock_httpx_module, mock_http_client) so tests can configure
    mock_http_client.post / mock_http_client.get as needed.

    The mock module's HTTPError is a real exception class so that
    ``except httpx.HTTPError`` in the production code works correctly.
    """
    mock_httpx = MagicMock()
    # Must be a real exception class for except clauses
    mock_httpx.HTTPError = _MockHTTPError
    mock_http_client = MagicMock()
    mock_http_client.__enter__ = Mock(return_value=mock_http_client)
    mock_http_client.__exit__ = Mock(return_value=False)
    mock_httpx.Client.return_value = mock_http_client
    return mock_httpx, mock_http_client


# ===========================================================================
# AriaUIClient tests
# ===========================================================================


class TestExtractCoordinates:
    """Tests for AriaUIClient._extract_coordinates static method."""

    def test_bracket_format(self):
        """Should parse '[523, 187]' format."""
        assert AriaUIClient._extract_coordinates("[523, 187]") == (523, 187)

    def test_paren_format(self):
        """Should parse '(523, 187)' format."""
        assert AriaUIClient._extract_coordinates("(523, 187)") == (523, 187)

    def test_bare_format(self):
        """Should parse '523, 187' format without brackets."""
        assert AriaUIClient._extract_coordinates("523, 187") == (523, 187)

    def test_embedded_in_text(self):
        """Should extract coordinates embedded in surrounding text."""
        assert AriaUIClient._extract_coordinates("some text [100, 200] more text") == (100, 200)

    def test_no_coordinates(self):
        """Should return None when no coordinates are present."""
        assert AriaUIClient._extract_coordinates("no coords") is None

    def test_empty_string(self):
        """Should return None for empty string."""
        assert AriaUIClient._extract_coordinates("") is None

    def test_whitespace_around_numbers(self):
        """Should handle extra whitespace around numbers."""
        assert AriaUIClient._extract_coordinates("[  42 ,  99  ]") == (42, 99)

    def test_zero_coordinates(self):
        """Should handle zero values."""
        assert AriaUIClient._extract_coordinates("[0, 0]") == (0, 0)

    def test_max_scale_coordinates(self):
        """Should handle max 1000 scale values."""
        assert AriaUIClient._extract_coordinates("[1000, 1000]") == (1000, 1000)


class TestParseAriaResponse:
    """Tests for AriaUIClient._parse_aria_response."""

    def setup_method(self):
        self.client = AriaUIClient(endpoint="http://localhost:8100")

    def test_valid_coords_absolute_conversion(self):
        """Should convert 0-1000 scale to absolute pixels using screenshot_shape."""
        context = _make_context(screenshot_shape=(1080, 1920))
        result = self.client._parse_aria_response("[500, 500]", context)

        assert result is not None
        # 500/1000 * 1920 = 960, 500/1000 * 1080 = 540
        assert result.x == 960
        assert result.y == 540
        assert result.confidence == 1.0

    def test_top_left_corner(self):
        """Should handle coordinates at origin."""
        context = _make_context(screenshot_shape=(1080, 1920))
        result = self.client._parse_aria_response("[0, 0]", context)

        assert result is not None
        assert result.x == 0
        assert result.y == 0

    def test_bottom_right_corner(self):
        """Should handle coordinates at max scale."""
        context = _make_context(screenshot_shape=(1080, 1920))
        result = self.client._parse_aria_response("[1000, 1000]", context)

        assert result is not None
        assert result.x == 1920
        assert result.y == 1080

    def test_out_of_range_above(self):
        """Should return None when coordinates exceed 1000."""
        context = _make_context(screenshot_shape=(1080, 1920))
        result = self.client._parse_aria_response("[1001, 500]", context)

        assert result is None

    def test_out_of_range_y_above(self):
        """Should return None when y coordinate exceeds 1000."""
        context = _make_context(screenshot_shape=(1080, 1920))
        result = self.client._parse_aria_response("[500, 1500]", context)

        assert result is None

    def test_fallback_to_1920x1080(self):
        """Should assume 1920x1080 when screenshot_shape is None."""
        context = _make_context(screenshot_shape=None)
        result = self.client._parse_aria_response("[500, 500]", context)

        assert result is not None
        # fallback: width=1920, height=1080
        # 500/1000 * 1920 = 960, 500/1000 * 1080 = 540
        assert result.x == 960
        assert result.y == 540

    def test_non_square_resolution(self):
        """Should handle non-standard resolutions correctly."""
        # screenshot_shape is (height, width)
        context = _make_context(screenshot_shape=(800, 1280))
        result = self.client._parse_aria_response("[250, 750]", context)

        assert result is not None
        # 250/1000 * 1280 = 320, 750/1000 * 800 = 600
        assert result.x == 320
        assert result.y == 600

    def test_unparseable_response(self):
        """Should return None for unparseable response."""
        context = _make_context()
        result = self.client._parse_aria_response("I cannot determine the location", context)

        assert result is None

    def test_response_description_field(self):
        """Should include grounding info in description."""
        context = _make_context(screenshot_shape=(1080, 1920))
        result = self.client._parse_aria_response("[523, 187]", context)

        assert result is not None
        assert "523" in result.description
        assert "187" in result.description


class TestAriaUIClientFindElement:
    """Tests for AriaUIClient.find_element with mocked httpx."""

    def setup_method(self):
        self.client = AriaUIClient(endpoint="http://test-server:8100")

    def test_successful_response(self):
        """Should return correct absolute coordinates on successful response."""
        mock_httpx, mock_http_client = _make_mock_httpx()
        mock_response = Mock()
        mock_response.json.return_value = _make_chat_response("[500, 500]")
        mock_response.raise_for_status = Mock()
        mock_http_client.post.return_value = mock_response

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            context = _make_context(screenshot_shape=(1080, 1920))
            result = self.client.find_element(FAKE_SCREENSHOT, context)

        assert result is not None
        assert result.x == 960
        assert result.y == 540

        # Verify correct endpoint was called
        mock_http_client.post.assert_called_once()
        call_args = mock_http_client.post.call_args
        assert call_args[0][0] == "http://test-server:8100/v1/chat/completions"

    def test_http_error_returns_none(self):
        """Should return None when HTTP request fails."""
        mock_httpx, mock_http_client = _make_mock_httpx()
        mock_http_client.post.side_effect = _MockHTTPError("Connection refused")

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            context = _make_context()
            result = self.client.find_element(FAKE_SCREENSHOT, context)

        assert result is None

    def test_malformed_json_returns_none(self):
        """Should return None when response JSON is malformed (missing choices key)."""
        mock_httpx, mock_http_client = _make_mock_httpx()
        mock_response = Mock()
        mock_response.json.return_value = {"error": "bad request"}
        mock_response.raise_for_status = Mock()
        mock_http_client.post.return_value = mock_response

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            context = _make_context()
            result = self.client.find_element(FAKE_SCREENSHOT, context)

        assert result is None

    def test_payload_contains_base64_screenshot(self):
        """Should encode screenshot as base64 in the payload."""
        mock_httpx, mock_http_client = _make_mock_httpx()
        mock_response = Mock()
        mock_response.json.return_value = _make_chat_response("[500, 500]")
        mock_response.raise_for_status = Mock()
        mock_http_client.post.return_value = mock_response

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            context = _make_context()
            self.client.find_element(FAKE_SCREENSHOT, context)

        call_kwargs = mock_http_client.post.call_args[1]
        payload = call_kwargs["json"]
        image_content = payload["messages"][0]["content"][1]
        expected_b64 = base64.b64encode(FAKE_SCREENSHOT).decode("utf-8")
        assert expected_b64 in image_content["image_url"]["url"]

    def test_payload_uses_correct_model(self):
        """Should use the default model name in the payload."""
        mock_httpx, mock_http_client = _make_mock_httpx()
        mock_response = Mock()
        mock_response.json.return_value = _make_chat_response("[500, 500]")
        mock_response.raise_for_status = Mock()
        mock_http_client.post.return_value = mock_response

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            context = _make_context()
            self.client.find_element(FAKE_SCREENSHOT, context)

        call_kwargs = mock_http_client.post.call_args[1]
        payload = call_kwargs["json"]
        assert payload["model"] == "Aria-UI/Aria-UI-base"

    def test_payload_prompt_contains_description(self):
        """Should include the element description in the prompt text."""
        mock_httpx, mock_http_client = _make_mock_httpx()
        mock_response = Mock()
        mock_response.json.return_value = _make_chat_response("[500, 500]")
        mock_response.raise_for_status = Mock()
        mock_http_client.post.return_value = mock_response

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            context = _make_context(description="Click the big red button")
            self.client.find_element(FAKE_SCREENSHOT, context)

        call_kwargs = mock_http_client.post.call_args[1]
        payload = call_kwargs["json"]
        prompt_text = payload["messages"][0]["content"][0]["text"]
        assert "Click the big red button" in prompt_text


class TestAriaUIClientIsAvailable:
    """Tests for AriaUIClient.is_available property."""

    def test_available_when_healthy(self):
        """Should return True when /health returns 200."""
        mock_httpx, mock_http_client = _make_mock_httpx()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_http_client.get.return_value = mock_response

        client = AriaUIClient(endpoint="http://localhost:8100")
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = client.is_available

        assert result is True
        mock_http_client.get.assert_called_once_with("http://localhost:8100/health")

    def test_unavailable_on_connection_error(self):
        """Should return False when connection fails."""
        mock_httpx, mock_http_client = _make_mock_httpx()
        mock_http_client.get.side_effect = ConnectionError("refused")

        client = AriaUIClient(endpoint="http://localhost:8100")
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = client.is_available

        assert result is False

    def test_unavailable_on_non_200(self):
        """Should return False when /health returns non-200."""
        mock_httpx, mock_http_client = _make_mock_httpx()
        mock_response = Mock()
        mock_response.status_code = 503
        mock_http_client.get.return_value = mock_response

        client = AriaUIClient(endpoint="http://localhost:8100")
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = client.is_available

        assert result is False


class TestAriaUIClientInit:
    """Tests for AriaUIClient initialization."""

    def test_default_endpoint(self):
        """Should use default endpoint."""
        client = AriaUIClient()
        assert client._endpoint == "http://localhost:8100"

    def test_custom_endpoint_strips_trailing_slash(self):
        """Should strip trailing slash from endpoint."""
        client = AriaUIClient(endpoint="http://myserver:9000/")
        assert client._endpoint == "http://myserver:9000"

    def test_custom_model(self):
        """Should allow model name override."""
        client = AriaUIClient(model="custom/model")
        assert client._model == "custom/model"

    def test_default_model(self):
        """Should use Aria-UI/Aria-UI-base as default model."""
        client = AriaUIClient()
        assert client._model == "Aria-UI/Aria-UI-base"

    def test_custom_timeout(self):
        """Should accept custom timeout."""
        client = AriaUIClient(timeout=30.0)
        assert client._timeout == 30.0


# ===========================================================================
# AriaUIContextClient tests
# ===========================================================================


class TestAriaUIContextClientInit:
    """Tests for AriaUIContextClient initialization."""

    def test_default_max_history(self):
        """Should default to max_history=3."""
        client = AriaUIContextClient()
        assert client._max_history == 3

    def test_custom_max_history(self):
        """Should accept custom max_history."""
        client = AriaUIContextClient(max_history=5)
        assert client._max_history == 5

    def test_default_model(self):
        """Should use context-aware model by default."""
        client = AriaUIContextClient()
        assert client._model == "Aria-UI/Aria-UI-context-aware"

    def test_inherits_from_aria_ui_client(self):
        """Should be a subclass of AriaUIClient."""
        assert issubclass(AriaUIContextClient, AriaUIClient)


class TestBuildContextMessages:
    """Tests for AriaUIContextClient._build_context_messages."""

    def setup_method(self):
        self.client = AriaUIContextClient(endpoint="http://localhost:8100", max_history=3)

    def test_no_history_just_current(self):
        """Should produce a single user message when history is empty."""
        context = _make_context(description="Click Submit")
        messages = self.client._build_context_messages(FAKE_SCREENSHOT, context, action_history=[])

        # Only the final grounding turn
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert any(
            "Click Submit" in part.get("text", "")
            for part in messages[0]["content"]
            if isinstance(part, dict)
        )

    def test_two_history_entries(self):
        """Should produce 2 user/assistant pairs + 1 current = 5 messages."""
        history = [
            (b"screenshot_1", "Clicked File menu"),
            (b"screenshot_2", "Clicked Save As"),
        ]
        context = _make_context(description="Click OK button")
        messages = self.client._build_context_messages(
            FAKE_SCREENSHOT, context, action_history=history
        )

        # 2 history entries x 2 messages (user+assistant) + 1 current user = 5
        assert len(messages) == 5

        # First history turn
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "done"

        # Second history turn
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[3]["content"] == "done"

        # Current grounding request
        assert messages[4]["role"] == "user"

    def test_truncates_to_max_history(self):
        """Should only include the most recent max_history entries."""
        client = AriaUIContextClient(max_history=2)
        history = [
            (b"old_1", "Old action 1"),
            (b"old_2", "Old action 2"),
            (b"recent_1", "Recent action 1"),
            (b"recent_2", "Recent action 2"),
        ]
        context = _make_context()
        messages = client._build_context_messages(FAKE_SCREENSHOT, context, action_history=history)

        # max_history=2 -> 2 pairs + 1 current = 5
        assert len(messages) == 5

        # Verify it used the most recent entries (Recent action 1 and 2)
        first_history_text = messages[0]["content"][1]["text"]
        assert first_history_text == "Recent action 1"

    def test_history_screenshots_are_base64_encoded(self):
        """Should base64-encode history screenshots."""
        history = [(b"hist_img", "action")]
        context = _make_context()
        messages = self.client._build_context_messages(
            FAKE_SCREENSHOT, context, action_history=history
        )

        expected_b64 = base64.b64encode(b"hist_img").decode("utf-8")
        image_url = messages[0]["content"][0]["image_url"]["url"]
        assert expected_b64 in image_url

    def test_current_screenshot_is_base64_encoded(self):
        """Should base64-encode the current screenshot in the final message."""
        context = _make_context()
        messages = self.client._build_context_messages(FAKE_SCREENSHOT, context, action_history=[])

        expected_b64 = base64.b64encode(FAKE_SCREENSHOT).decode("utf-8")
        image_url = messages[0]["content"][0]["image_url"]["url"]
        assert expected_b64 in image_url

    def test_history_image_url_has_data_uri_prefix(self):
        """Should use data:image/png;base64, prefix in image URLs."""
        history = [(b"img", "action")]
        context = _make_context()
        messages = self.client._build_context_messages(
            FAKE_SCREENSHOT, context, action_history=history
        )

        image_url = messages[0]["content"][0]["image_url"]["url"]
        assert image_url.startswith("data:image/png;base64,")

    def test_max_history_one(self):
        """Should work correctly with max_history=1."""
        client = AriaUIContextClient(max_history=1)
        history = [
            (b"first", "First action"),
            (b"second", "Second action"),
            (b"third", "Third action"),
        ]
        context = _make_context()
        messages = client._build_context_messages(FAKE_SCREENSHOT, context, action_history=history)

        # 1 pair + 1 current = 3
        assert len(messages) == 3
        # Should keep only the last entry
        assert messages[0]["content"][1]["text"] == "Third action"


class TestAriaUIContextClientFindElement:
    """Tests for AriaUIContextClient.find_element dispatch logic."""

    def setup_method(self):
        self.client = AriaUIContextClient(endpoint="http://localhost:8100")

    @patch.object(AriaUIContextClient, "find_element_with_history")
    def test_delegates_to_history_when_action_history_present(self, mock_find_hist):
        """Should call find_element_with_history when action_history is in context."""
        mock_find_hist.return_value = ElementLocation(x=100, y=200, confidence=1.0)

        history = [(b"prev_screenshot", "Clicked button")]
        context = _make_context(additional_context={"action_history": history})

        result = self.client.find_element(FAKE_SCREENSHOT, context)

        mock_find_hist.assert_called_once_with(FAKE_SCREENSHOT, context, history)
        assert result is not None
        assert result.x == 100

    @patch.object(AriaUIClient, "find_element")
    def test_falls_back_to_base_when_no_history(self, mock_base_find):
        """Should fall back to base AriaUIClient.find_element with no history."""
        mock_base_find.return_value = ElementLocation(x=50, y=75, confidence=1.0)

        context = _make_context(additional_context={})

        result = self.client.find_element(FAKE_SCREENSHOT, context)

        mock_base_find.assert_called_once_with(FAKE_SCREENSHOT, context)
        assert result is not None
        assert result.x == 50

    @patch.object(AriaUIClient, "find_element")
    def test_falls_back_to_base_when_empty_history(self, mock_base_find):
        """Should fall back to base when action_history is an empty list."""
        mock_base_find.return_value = None

        context = _make_context(additional_context={"action_history": []})

        result = self.client.find_element(FAKE_SCREENSHOT, context)

        mock_base_find.assert_called_once()
        assert result is None

    @patch.object(AriaUIClient, "find_element")
    def test_falls_back_when_action_history_key_missing(self, mock_base_find):
        """Should fall back to base when additional_context has no action_history key."""
        mock_base_find.return_value = ElementLocation(x=10, y=20, confidence=0.9)

        context = _make_context(additional_context={"some_other_key": "value"})

        result = self.client.find_element(FAKE_SCREENSHOT, context)

        mock_base_find.assert_called_once()
        assert result.x == 10


class TestAriaUIContextClientFindElementWithHistory:
    """Tests for AriaUIContextClient.find_element_with_history with mocked httpx."""

    def setup_method(self):
        self.client = AriaUIContextClient(endpoint="http://test-server:8100")

    def test_successful_multi_turn_response(self):
        """Should send multi-turn messages and parse response."""
        mock_httpx, mock_http_client = _make_mock_httpx()
        mock_response = Mock()
        mock_response.json.return_value = _make_chat_response("[300, 700]")
        mock_response.raise_for_status = Mock()
        mock_http_client.post.return_value = mock_response

        history = [(b"prev", "Clicked menu")]
        context = _make_context(screenshot_shape=(1080, 1920))

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = self.client.find_element_with_history(FAKE_SCREENSHOT, context, history)

        assert result is not None
        # 300/1000 * 1920 = 576, 700/1000 * 1080 = 756
        assert result.x == 576
        assert result.y == 756

        # Verify multi-turn messages were sent
        call_kwargs = mock_http_client.post.call_args[1]
        payload = call_kwargs["json"]
        messages = payload["messages"]
        # 1 history pair (user+assistant) + 1 current = 3 messages
        assert len(messages) == 3
        assert payload["model"] == "Aria-UI/Aria-UI-context-aware"

    def test_http_error_returns_none(self):
        """Should return None on HTTP error."""
        mock_httpx, mock_http_client = _make_mock_httpx()
        mock_http_client.post.side_effect = _MockHTTPError("timeout")

        history = [(b"prev", "action")]
        context = _make_context()

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = self.client.find_element_with_history(FAKE_SCREENSHOT, context, history)

        assert result is None

    def test_malformed_response_returns_none(self):
        """Should return None when response JSON has missing keys."""
        mock_httpx, mock_http_client = _make_mock_httpx()
        mock_response = Mock()
        mock_response.json.return_value = {"unexpected": "format"}
        mock_response.raise_for_status = Mock()
        mock_http_client.post.return_value = mock_response

        history = [(b"prev", "action")]
        context = _make_context()

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = self.client.find_element_with_history(FAKE_SCREENSHOT, context, history)

        assert result is None

    def test_calls_correct_endpoint(self):
        """Should POST to /v1/chat/completions on the configured endpoint."""
        mock_httpx, mock_http_client = _make_mock_httpx()
        mock_response = Mock()
        mock_response.json.return_value = _make_chat_response("[500, 500]")
        mock_response.raise_for_status = Mock()
        mock_http_client.post.return_value = mock_response

        history = [(b"prev", "action")]
        context = _make_context()

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            self.client.find_element_with_history(FAKE_SCREENSHOT, context, history)

        call_args = mock_http_client.post.call_args
        assert call_args[0][0] == "http://test-server:8100/v1/chat/completions"
