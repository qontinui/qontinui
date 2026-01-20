"""
Tests for hybrid_extractor module.

Tests combined DOM + screenshot context extraction for LLMs.
"""

from qontinui.extraction.web.hybrid_extractor import (
    HybridContext,
    PageState,
    build_llm_prompt,
)
from qontinui.extraction.web.models import BoundingBox, InteractiveElement


def create_test_element(id: str, text: str) -> InteractiveElement:
    """Helper to create test elements."""
    return InteractiveElement(
        id=id,
        bbox=BoundingBox(x=0, y=0, width=100, height=50),
        tag_name="button",
        element_type="button",
        screenshot_id="test_screen",
        selector=f"button#{id}",
        text=text,
    )


class TestHybridContext:
    """Tests for HybridContext dataclass."""

    def test_create_context(self) -> None:
        """Test creating a HybridContext."""
        elements = [create_test_element("btn1", "Submit")]

        context = HybridContext(
            url="https://example.com",
            title="Test Page",
            viewport=(1920, 1080),
            screenshot_base64="base64encodedimage",
            elements=elements,
            elements_formatted="[0]<button>Submit</button>",
        )

        assert context.url == "https://example.com"
        assert context.title == "Test Page"
        assert len(context.elements) == 1
        assert context.frame_count == 1

    def test_context_with_scroll_info(self) -> None:
        """Test context with scroll information."""
        context = HybridContext(
            url="https://example.com",
            title="Test Page",
            viewport=(1920, 1080),
            screenshot_base64="",
            scroll_x=0,
            scroll_y=500,
            scroll_height=2000,
            viewport_height=1080,
        )

        assert context.scroll_y == 500
        assert context.scroll_height == 2000

    def test_context_with_iframes(self) -> None:
        """Test context with iframe info."""
        context = HybridContext(
            url="https://example.com",
            title="Test Page",
            viewport=(1920, 1080),
            screenshot_base64="",
            frame_count=3,
            has_iframes=True,
        )

        assert context.frame_count == 3
        assert context.has_iframes is True

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        context = HybridContext(
            url="https://example.com",
            title="Test Page",
            viewport=(1920, 1080),
            screenshot_base64="shortbase64",
            elements_formatted="[0]<button>Submit</button>",
        )

        data = context.to_dict()

        assert data["url"] == "https://example.com"
        assert data["title"] == "Test Page"
        assert data["viewport"] == [1920, 1080]

    def test_to_llm_message_with_screenshot(self) -> None:
        """Test converting to LLM message format."""
        context = HybridContext(
            url="https://example.com",
            title="Test Page",
            viewport=(1920, 1080),
            screenshot_base64="base64data",
            screenshot_format="jpeg",
            elements_formatted="[0]<button>Submit</button>",
        )

        message = context.to_llm_message(include_screenshot=True)

        assert "text" in message
        assert "image" in message
        assert message["image"]["type"] == "jpeg"
        assert "Test Page" in message["text"]

    def test_to_llm_message_without_screenshot(self) -> None:
        """Test LLM message without screenshot."""
        context = HybridContext(
            url="https://example.com",
            title="Test Page",
            viewport=(1920, 1080),
            screenshot_base64="base64data",
            elements_formatted="[0]<button>Submit</button>",
        )

        message = context.to_llm_message(include_screenshot=False)

        assert "text" in message
        assert "image" not in message

    def test_to_llm_message_with_accessibility(self) -> None:
        """Test LLM message includes accessibility tree."""
        context = HybridContext(
            url="https://example.com",
            title="Test Page",
            viewport=(1920, 1080),
            screenshot_base64="",
            elements_formatted="[0]<button>Submit</button>",
            accessibility_tree_text='button "Submit"',
        )

        message = context.to_llm_message(include_screenshot=False)

        assert "Accessibility Tree" in message["text"]
        assert "Submit" in message["text"]


class TestPageState:
    """Tests for PageState dataclass."""

    def test_create_state(self) -> None:
        """Test creating a PageState."""
        context = HybridContext(
            url="https://example.com",
            title="Test Page",
            viewport=(1920, 1080),
            screenshot_base64="",
        )

        state = PageState(
            context=context,
            timestamp=1234567890.0,
            state_hash="abc123",
        )

        assert state.context.url == "https://example.com"
        assert state.timestamp == 1234567890.0
        assert state.state_hash == "abc123"

    def test_to_dict(self) -> None:
        """Test serialization."""
        context = HybridContext(
            url="https://example.com",
            title="Test",
            viewport=(1920, 1080),
            screenshot_base64="",
        )

        state = PageState(
            context=context,
            timestamp=1234567890.0,
            state_hash="abc123",
        )

        data = state.to_dict()

        assert "context" in data
        assert data["timestamp"] == 1234567890.0
        assert data["state_hash"] == "abc123"


class TestBuildLLMPrompt:
    """Tests for build_llm_prompt function."""

    def test_build_prompt_basic(self) -> None:
        """Test building a basic LLM prompt."""
        context = HybridContext(
            url="https://example.com",
            title="Test Page",
            viewport=(1920, 1080),
            screenshot_base64="base64data",
            elements_formatted="[0]<button>Submit</button>",
        )

        prompt = build_llm_prompt(
            context=context,
            instruction="Click the submit button",
        )

        assert "system" in prompt
        assert "user" in prompt
        assert "Click the submit button" in prompt["user"]["text"]

    def test_build_prompt_system_content(self) -> None:
        """Test that system prompt contains instructions."""
        context = HybridContext(
            url="https://example.com",
            title="Test",
            viewport=(1920, 1080),
            screenshot_base64="",
            elements_formatted="",
        )

        prompt = build_llm_prompt(context, "test instruction")

        system = prompt["system"]
        assert "web automation" in system.lower()
        assert "element" in system.lower()

    def test_build_prompt_without_screenshot(self) -> None:
        """Test building prompt without screenshot."""
        context = HybridContext(
            url="https://example.com",
            title="Test",
            viewport=(1920, 1080),
            screenshot_base64="data",
            elements_formatted="[0]<button>OK</button>",
        )

        prompt = build_llm_prompt(
            context,
            "Click OK",
            include_screenshot=False,
        )

        # User message should not have image
        assert "image" not in prompt["user"]


class TestHybridExtractorConfig:
    """Tests for HybridExtractor configuration."""

    def test_default_config(self) -> None:
        """Test default extractor configuration."""
        from qontinui.extraction.web.hybrid_extractor import HybridExtractor

        extractor = HybridExtractor()

        assert extractor.include_accessibility is True
        assert extractor.include_shadow_dom is True
        assert extractor.include_iframes is True
        assert extractor.screenshot_format == "jpeg"

    def test_custom_config(self) -> None:
        """Test custom extractor configuration."""
        from qontinui.extraction.web.hybrid_extractor import HybridExtractor

        extractor = HybridExtractor(
            include_accessibility=False,
            include_iframes=False,
            screenshot_format="png",
            max_elements=100,
        )

        assert extractor.include_accessibility is False
        assert extractor.include_iframes is False
        assert extractor.screenshot_format == "png"
        assert extractor.max_elements == 100


class TestStateTracker:
    """Tests for StateTracker class."""

    def test_tracker_init(self) -> None:
        """Test StateTracker initialization."""
        from qontinui.extraction.web.hybrid_extractor import StateTracker

        tracker = StateTracker()

        assert tracker.states == []

    def test_is_same_state(self) -> None:
        """Test comparing states by hash."""
        from qontinui.extraction.web.hybrid_extractor import StateTracker

        tracker = StateTracker()

        context1 = HybridContext(
            url="https://example.com",
            title="Test",
            viewport=(1920, 1080),
            screenshot_base64="",
        )
        context2 = HybridContext(
            url="https://example.com/page2",
            title="Test 2",
            viewport=(1920, 1080),
            screenshot_base64="",
        )

        state1 = PageState(context=context1, timestamp=1.0, state_hash="hash1")
        state2 = PageState(context=context2, timestamp=2.0, state_hash="hash1")
        state3 = PageState(context=context2, timestamp=3.0, state_hash="hash2")

        # Same hash = same state
        assert tracker.is_same_state(state1, state2) is True
        # Different hash = different state
        assert tracker.is_same_state(state1, state3) is False

    def test_get_state_diff(self) -> None:
        """Test state diff calculation."""
        from qontinui.extraction.web.hybrid_extractor import StateTracker

        tracker = StateTracker()

        context1 = HybridContext(
            url="https://example.com",
            title="Page 1",
            viewport=(1920, 1080),
            screenshot_base64="",
            elements=[create_test_element("btn1", "Submit")],
        )
        context2 = HybridContext(
            url="https://example.com/page2",
            title="Page 2",
            viewport=(1920, 1080),
            screenshot_base64="",
            elements=[
                create_test_element("btn1", "Submit"),
                create_test_element("btn2", "Cancel"),
            ],
        )

        state1 = PageState(context=context1, timestamp=1.0, state_hash="hash1")
        state2 = PageState(context=context2, timestamp=2.0, state_hash="hash2")

        diff = tracker.get_state_diff(state1, state2)

        assert diff["url_changed"] is True
        assert diff["title_changed"] is True
        assert diff["element_count_changed"] is True
        assert diff["state_hash_changed"] is True
        assert diff["elements_before"] == 1
        assert diff["elements_after"] == 2

    def test_clear_history(self) -> None:
        """Test clearing state history."""
        from qontinui.extraction.web.hybrid_extractor import StateTracker

        tracker = StateTracker()

        context = HybridContext(
            url="https://example.com",
            title="Test",
            viewport=(1920, 1080),
            screenshot_base64="",
        )
        tracker.states.append(PageState(context=context, timestamp=1.0))

        assert len(tracker.states) == 1

        tracker.clear_history()

        assert len(tracker.states) == 0
