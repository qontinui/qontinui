"""Tests for UI-TARS inference providers."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from qontinui.extraction.runtime.uitars.config import UITARSSettings  # noqa: E402
from qontinui.extraction.runtime.uitars.models import (  # noqa: E402
    UITARSActionType,
    UITARSInferenceRequest,
)
from qontinui.extraction.runtime.uitars.provider import (  # noqa: E402
    HuggingFaceEndpointProvider,
    LocalTransformersProvider,
    VLLMProvider,
    create_provider,
)


class TestParseOutput:
    """Test output parsing for all providers."""

    def setup_method(self):
        """Set up test fixtures."""
        self.settings = UITARSSettings()
        self.provider = LocalTransformersProvider(self.settings)

    def test_parse_click_action(self):
        """Test parsing click action."""
        raw_output = """Thought: I see a submit button in the lower right corner.
Action: click(500, 300)"""

        thought, action = self.provider.parse_output(raw_output)

        assert "submit button" in thought.reasoning.lower()
        assert action.action_type == UITARSActionType.CLICK
        assert action.x == 500
        assert action.y == 300

    def test_parse_double_click_action(self):
        """Test parsing double click action."""
        raw_output = """Thought: Need to double click to open the file.
Action: double_click(250, 150)"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.DOUBLE_CLICK
        assert action.x == 250
        assert action.y == 150

    def test_parse_right_click_action(self):
        """Test parsing right click action."""
        raw_output = """Thought: Opening context menu.
Action: right_click(400, 200)"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.RIGHT_CLICK
        assert action.x == 400
        assert action.y == 200

    def test_parse_type_action_with_coords(self):
        """Test parsing type action with coordinates."""
        raw_output = """Thought: Typing in the search box.
Action: type(300, 100, 'hello world')"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.TYPE
        assert action.x == 300
        assert action.y == 100
        assert action.text == "hello world"

    def test_parse_type_action_text_only(self):
        """Test parsing type action with text only."""
        raw_output = """Thought: Typing text.
Action: type('test input')"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.TYPE
        assert action.text == "test input"

    def test_parse_scroll_with_direction_only(self):
        """Test parsing scroll action with direction only."""
        raw_output = """Thought: Scrolling down to see more content.
Action: scroll(down)"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.SCROLL
        assert action.scroll_direction == "down"
        assert action.scroll_amount == 100  # Default

    def test_parse_scroll_with_coords_and_amount(self):
        """Test parsing scroll action with coordinates and amount."""
        raw_output = """Thought: Scrolling in the list area.
Action: scroll(400, 300, up, 200)"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.SCROLL
        assert action.x == 400
        assert action.y == 300
        assert action.scroll_direction == "up"
        assert action.scroll_amount == 200

    def test_parse_hover_action(self):
        """Test parsing hover action."""
        raw_output = """Thought: Hovering over the menu item.
Action: hover(350, 175)"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.HOVER
        assert action.x == 350
        assert action.y == 175

    def test_parse_drag_action(self):
        """Test parsing drag action."""
        raw_output = """Thought: Dragging the slider.
Action: drag(100, 200, 300, 200)"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.DRAG
        assert action.x == 100
        assert action.y == 200
        assert action.end_x == 300
        assert action.end_y == 200

    def test_parse_hotkey_action(self):
        """Test parsing hotkey action."""
        raw_output = """Thought: Saving the file.
Action: hotkey(ctrl, s)"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.HOTKEY
        assert action.keys == ["ctrl", "s"]

    def test_parse_wait_action(self):
        """Test parsing wait action."""
        raw_output = """Thought: Waiting for page to load.
Action: wait(2.5)"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.WAIT
        assert action.duration == 2.5

    def test_parse_done_action(self):
        """Test parsing done action."""
        raw_output = """Thought: Task is complete.
Action: done()"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.DONE

    def test_parse_click_with_float_coords(self):
        """Test parsing click with float coordinates."""
        raw_output = """Thought: Clicking button.
Action: click(123.5, 456.7)"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.CLICK
        assert action.x == 123
        assert action.y == 456

    def test_parse_empty_thought(self):
        """Test parsing output with empty thought."""
        raw_output = """Action: click(100, 200)"""

        thought, action = self.provider.parse_output(raw_output)

        assert thought.reasoning == ""
        assert action.action_type == UITARSActionType.CLICK

    def test_parse_multiline_thought(self):
        """Test parsing multi-line thought."""
        raw_output = """Thought: First, I observe the screen.
Then I see a button.
Finally, I will click it.
Action: click(100, 200)"""

        thought, action = self.provider.parse_output(raw_output)

        assert "First" in thought.reasoning
        assert "button" in thought.reasoning
        assert action.action_type == UITARSActionType.CLICK

    def test_parse_case_insensitive(self):
        """Test case-insensitive parsing."""
        raw_output = """THOUGHT: Testing case.
ACTION: CLICK(100, 200)"""

        thought, action = self.provider.parse_output(raw_output)

        assert thought.reasoning == "Testing case."
        assert action.action_type == UITARSActionType.CLICK

    def test_parse_unknown_action_defaults_to_wait(self):
        """Test unknown action type defaults to WAIT."""
        raw_output = """Thought: Unknown action.
Action: unknown_action(100, 200)"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.WAIT

    def test_parse_missing_action_defaults_to_wait(self):
        """Test missing action defaults to WAIT."""
        raw_output = """Thought: No action specified."""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.WAIT

    def test_parse_invalid_coordinates_handled(self):
        """Test invalid coordinates are handled gracefully."""
        raw_output = """Thought: Bad coords.
Action: click(abc, def)"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.CLICK
        assert action.x is None
        assert action.y is None

    def test_raw_output_stored(self):
        """Test raw output is stored in action."""
        raw_output = """Thought: Test.
Action: click(100, 200)"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.raw_output == raw_output


class TestImageToBase64:
    """Test image to base64 conversion."""

    def setup_method(self):
        """Set up test fixtures."""
        self.settings = UITARSSettings()
        self.provider = LocalTransformersProvider(self.settings)

    def test_convert_rgb_image(self):
        """Test converting RGB numpy array to base64."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[50, 50] = [255, 0, 0]  # Red pixel

        base64_str = self.provider._image_to_base64(image)

        assert isinstance(base64_str, str)
        assert len(base64_str) > 0

    def test_convert_larger_image(self):
        """Test converting larger image."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        base64_str = self.provider._image_to_base64(image)

        assert isinstance(base64_str, str)


class TestCreateProvider:
    """Test provider factory function."""

    def test_create_cloud_provider(self):
        """Test creating cloud provider."""
        settings = UITARSSettings(
            provider="cloud",
            huggingface_endpoint="https://test.endpoint.com",
        )

        provider = create_provider(settings)

        assert isinstance(provider, HuggingFaceEndpointProvider)
        assert provider.settings == settings

    def test_create_local_transformers_provider(self):
        """Test creating local transformers provider."""
        settings = UITARSSettings(provider="local_transformers")

        provider = create_provider(settings)

        assert isinstance(provider, LocalTransformersProvider)

    def test_create_vllm_provider(self):
        """Test creating vLLM provider."""
        settings = UITARSSettings(provider="local_vllm")

        provider = create_provider(settings)

        assert isinstance(provider, VLLMProvider)

    def test_create_invalid_provider_raises_error(self):
        """Test that invalid provider raises ValueError."""
        settings = UITARSSettings()
        settings.provider = "invalid_provider"  # type: ignore[assignment]

        try:
            create_provider(settings)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown provider" in str(e)


class TestHuggingFaceEndpointProvider:
    """Test HuggingFace endpoint provider."""

    def setup_method(self):
        """Set up test fixtures."""
        self.settings = UITARSSettings(
            provider="cloud",
            huggingface_endpoint="https://test.endpoint.com",
            huggingface_api_token="hf_test_token",
        )
        self.provider = HuggingFaceEndpointProvider(self.settings)

    def test_not_available_without_endpoint(self):
        """Test provider not available without endpoint configured."""
        settings = UITARSSettings(provider="cloud")
        provider = HuggingFaceEndpointProvider(settings)

        assert provider.is_available() is False

    def test_not_available_without_initialization(self):
        """Test provider not available before initialization."""
        assert self.provider._initialized is False
        assert self.provider.is_available() is False

    def test_initialize_creates_client(self):
        """Test initialize creates HTTP client."""
        import sys

        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_httpx.Client.return_value = mock_client_instance
        mock_httpx.Timeout = MagicMock()

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            # Need fresh provider to use mock
            provider = HuggingFaceEndpointProvider(self.settings)
            provider.initialize()

            assert provider._initialized is True
            mock_httpx.Client.assert_called_once()

    def test_infer_returns_result(self):
        """Test inference returns parsed result."""
        import sys

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "generated_text": "Thought: Test\nAction: click(100, 200)"
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        mock_httpx = MagicMock()
        mock_httpx.Client.return_value = mock_client
        mock_httpx.Timeout = MagicMock()

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            # Need fresh provider to use mock
            provider = HuggingFaceEndpointProvider(self.settings)
            provider.initialize()

            image = np.zeros((100, 100, 3), dtype=np.uint8)
            request = UITARSInferenceRequest(image=image, prompt="Click button")

            result = provider.infer(request)

            assert result.success is True
            assert result.provider == "huggingface_endpoint"


class TestLocalTransformersProvider:
    """Test local transformers provider."""

    def setup_method(self):
        """Set up test fixtures."""
        self.settings = UITARSSettings(
            provider="local_transformers",
            model_size="2B",
            quantization="int4",
        )
        self.provider = LocalTransformersProvider(self.settings)

    def test_not_available_before_init(self):
        """Test provider not available before initialization."""
        assert self.provider.is_available() is False

    def test_available_after_model_loaded(self):
        """Test provider available after model loaded."""
        self.provider._initialized = True
        self.provider._model = MagicMock()

        assert self.provider.is_available() is True


class TestVLLMProvider:
    """Test vLLM provider."""

    def setup_method(self):
        """Set up test fixtures."""
        self.settings = UITARSSettings(
            provider="local_vllm",
            vllm_server_url="http://localhost:8000",
        )
        self.provider = VLLMProvider(self.settings)

    def test_not_available_before_init(self):
        """Test provider not available before initialization."""
        assert self.provider.is_available() is False

    def test_initialize_creates_client(self):
        """Test initialize creates HTTP client."""
        import sys

        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_httpx.Client.return_value = mock_client_instance
        mock_httpx.Timeout = MagicMock()

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            # Need fresh provider to use mock
            provider = VLLMProvider(self.settings)
            provider.initialize()

            assert provider._initialized is True
            mock_httpx.Client.assert_called_once()

    def test_infer_uses_openai_format(self):
        """Test inference uses OpenAI-compatible format."""
        import sys

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Thought: Test\nAction: click(100, 200)"}}],
            "usage": {"completion_tokens": 10},
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        mock_httpx = MagicMock()
        mock_httpx.Client.return_value = mock_client
        mock_httpx.Timeout = MagicMock()

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            provider = VLLMProvider(self.settings)
            provider.initialize()

            image = np.zeros((100, 100, 3), dtype=np.uint8)
            request = UITARSInferenceRequest(image=image, prompt="Test")

            result = provider.infer(request)

            # Check OpenAI endpoint was called
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "/v1/chat/completions"
            assert result.provider == "vllm"


class TestActionParsingEdgeCases:
    """Test edge cases in action parsing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.settings = UITARSSettings()
        self.provider = LocalTransformersProvider(self.settings)

    def test_parse_action_with_quoted_params(self):
        """Test parsing action with quoted parameters."""
        raw_output = """Thought: Typing text.
Action: type("hello world")"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.TYPE
        assert action.text == "hello world"

    def test_parse_action_with_single_quotes(self):
        """Test parsing action with single quotes."""
        raw_output = """Thought: Test.
Action: type('test text')"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.text == "test text"

    def test_parse_scroll_direction_variants(self):
        """Test parsing various scroll directions."""
        for direction in ["up", "down", "left", "right"]:
            raw_output = f"""Thought: Scrolling.
Action: scroll({direction})"""

            thought, action = self.provider.parse_output(raw_output)

            assert action.action_type == UITARSActionType.SCROLL
            assert action.scroll_direction == direction

    def test_parse_hotkey_multiple_keys(self):
        """Test parsing hotkey with multiple keys."""
        raw_output = """Thought: Copy all.
Action: hotkey(ctrl, shift, a)"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.HOTKEY
        assert action.keys == ["ctrl", "shift", "a"]

    def test_parse_empty_hotkey(self):
        """Test parsing hotkey with no keys."""
        raw_output = """Thought: Empty hotkey.
Action: hotkey()"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.HOTKEY
        assert action.keys == []

    def test_parse_wait_without_duration(self):
        """Test parsing wait without duration."""
        raw_output = """Thought: Wait.
Action: wait()"""

        thought, action = self.provider.parse_output(raw_output)

        assert action.action_type == UITARSActionType.WAIT
        # Duration should be None when no params provided

    def test_parse_extra_whitespace(self):
        """Test parsing with extra whitespace."""
        raw_output = """  Thought:   Lots of spaces
Action:   click(  100  ,  200  )  """

        thought, action = self.provider.parse_output(raw_output)

        assert "Lots of spaces" in thought.reasoning
        assert action.action_type == UITARSActionType.CLICK
